/* -----------------------------------------------------------------
FILE:    dstring.c
DESCRIPTION:This file contains the routines for manipulating dynamic strings.

Copyright 2020 The ngspice team
3 - Clause BSD license
(see COPYING or https://opensource.org/licenses/BSD-3-Clause)
Author: Jim Monte
----------------------------------------------------------------- */
#include <ctype.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ngspice/dstring.h"


static int ds_reserve_internal(DSTRING *p_ds,
        size_t n_byte_alloc_opt, size_t n_byte_alloc_min);

/* Instantiations of dstring functions */
extern inline int ds_cat_str(DSTRING *p_ds, const char *sz);
extern inline int ds_cat_char(DSTRING *p_ds, char c);
extern inline int ds_cat_ds(DSTRING *p_ds_dst, const DSTRING *p_ds_src);
extern inline int ds_cat_mem(DSTRING *p_ds, const char *p_src, size_t n_char);
extern inline int ds_set_length(DSTRING *p_ds, size_t length);
extern inline void ds_clear(DSTRING *p_ds);
extern inline char *ds_free_move(DSTRING *p_ds, unsigned int opt);
extern inline char *ds_get_buf(DSTRING *p_ds);
extern inline size_t ds_get_length(const DSTRING *p_ds);
extern inline size_t ds_get_buf_size(const DSTRING *p_ds);


/* This function initalizes a dstring using *p_buf as the initial backing
 *
 * Parameters
 * p_buf: Inital buffer backing the dstring
 * length_string: Length of string in the initial buffer
 * n_byte_data: Length of initial buffer. Must be at least 1
 * type_buffer: Type of buffer providing initial backing
 *
 * Return codes
 * DS_E_OK: Init OK
 * DS_E_INVALID: n_byte_data = 0 length_string too long,
 *      or unknown buffer type
 */
int ds_init(DSTRING *p_ds, char *p_buf, size_t length_string,
        size_t n_byte_buf, ds_buf_type_t type_buffer)
{
    /* Validate buffer size */
    if (n_byte_buf == 0) {
        return DS_E_INVALID;
    }

    /* Set current buffer */
    p_ds->p_buf = p_buf;

    /* Set size of current string >= rather than > because this function
     * adds a terminating null */
    if (length_string >= n_byte_buf) {
        return DS_E_INVALID;
    }

    p_ds->n_byte_alloc = n_byte_buf;
    p_ds->length = length_string;
    p_ds->p_buf[length_string] = '\0';

    /* Set stack buffer */
    if (type_buffer == ds_buf_type_stack) {
        p_ds->p_stack_buf = p_buf;
        p_ds->n_byte_stack_buf = n_byte_buf;
    }
    else if (type_buffer == ds_buf_type_heap) {
        p_ds->p_stack_buf = (char *) NULL;
        p_ds->n_byte_stack_buf = 0;
    }
    else { /* unknown buffer type */
        return DS_E_INVALID;
    }

    return DS_E_OK;
} /* end of function ds_init */



/* This function frees all memory used by the dstring. After calling this
 * function, the dstring should not be used again. */
void ds_free(DSTRING *p_ds)
{
    if (p_ds->p_buf != p_ds->p_stack_buf) {
        txfree((void *) p_ds->p_buf);
    }
} /* end of function ds_free */



/* Concatenate string */
int ds_cat_str_case(DSTRING *p_ds, const char *sz, ds_case_t case_type)
{
    return ds_cat_mem_case(p_ds, sz, strlen(sz), case_type);
} /* end of function ds_cat_str_case */



/* Concatenate character */
int ds_cat_char_case(DSTRING *p_ds, char c, ds_case_t case_type)
{
    return ds_cat_mem_case(p_ds, &c, 1, case_type);
} /* end of function ds_cat_char_case */



/* Concatenate another dstring */
int ds_cat_ds_case(DSTRING *p_ds_dst, const DSTRING *p_ds_src,
        ds_case_t case_type)
{
    return ds_cat_mem_case(p_ds_dst, p_ds_src->p_buf, p_ds_src->length,
            case_type);
} /* end of function ds_cat_ds_case */



/* General concatenation of a memory buffer. A terminating null is added. */
int ds_cat_mem_case(DSTRING *p_ds, const char *p_src, size_t n_char,
        ds_case_t type_case)
{
    /* Resize buffer if necessary. Double required size, if available,
     * to reduce the number of allocations */
    const size_t length_new = p_ds->length + n_char;
    const size_t n_byte_needed = length_new + 1;
    if (n_byte_needed > p_ds->n_byte_alloc) {
        if (ds_reserve_internal(p_ds,
                2 * n_byte_needed, n_byte_needed) == DS_E_NO_MEMORY) {
            return DS_E_NO_MEMORY;
        }
    }

    /* For "as-is" can simply memcpy */
    if (type_case == ds_case_as_is) {
        char *p_dst = p_ds->p_buf + p_ds->length;
        (void) memcpy(p_dst, p_src, n_char);
        p_dst += n_char;
        *p_dst = '\0';
        p_ds->length = length_new;
        return DS_E_OK;
    }

    /* For lowercasing, work char by char */
    if (type_case == ds_case_lower) {
        char *p_dst = p_ds->p_buf + p_ds->length;
        char *p_dst_end = p_dst + n_char;
        for ( ; p_dst < p_dst_end; p_dst++, p_src++) {
            *p_dst = (char) tolower(*p_src);
        }
        *p_dst_end = '\0';
        p_ds->length = length_new;
        return DS_E_OK;
    }

    /* Uppercasing done like lowercasing. Note that it would be possible to
     * use a function pointer and select either tolower() or toupper() based
     * on type_case, but doing so may degrade performance by inhibiting
     * inlining. */
    if (type_case == ds_case_upper) {
        char *p_dst = p_ds->p_buf + p_ds->length;
        char *p_dst_end = p_dst + n_char;
        for ( ; p_dst < p_dst_end; p_dst++, p_src++) {
            *p_dst = (char) toupper(*p_src);
        }
        *p_dst_end = '\0';
        p_ds->length = length_new;
        return DS_E_OK;
    }

    return DS_E_INVALID; /* unknown case type */
} /* end of function ds_cat_mem_case */



/* Ensure minimum internal buffer size */
int ds_reserve(DSTRING *p_ds, size_t n_byte_alloc)
{
    /* Return if buffer already large enough */
    if (p_ds->n_byte_alloc >= n_byte_alloc) {
        return DS_E_OK;
    }

    return ds_reserve_internal(p_ds, n_byte_alloc, 0);
} /* end of function ds_reserve */



/* This function resizes the buffer for the string and handles freeing
 * the original alloction, if necessary. It is assumed that the requested
 * size or sizes are larger than the current size.
 *
 * Parameters
 * p_ds: Dstring pointer
 * n_byte_alloc_opt: Optimal alloction amount
 * n_byte_alloc_min: Absolute minimum allocation amount or 0 if no
 *      smaller amount can be allocated
 *
 * Return codes
 * DS_E_OK: At least the minimum allocation was performed
 * DS_E_NO_MEMORY: Unable to resize the buffer */
static int ds_reserve_internal(DSTRING *p_ds,
        size_t n_byte_alloc_opt, size_t n_byte_alloc_min)
{
    size_t n_byte_alloc = n_byte_alloc_opt;
    /* Allocate. First try (larger) optimal size, and gradually fall back
     * to min size if that fails and one was provided. */
    char * p_buf_new;
    if (n_byte_alloc_min == 0) {
        n_byte_alloc_min = n_byte_alloc_opt;
    }
    for ( ; ; ) {
        if ((p_buf_new = (char *) malloc(n_byte_alloc)) != (char *) NULL) {
            break; /* Allocated OK */
        }

        if (n_byte_alloc == n_byte_alloc_min) { /* min alloc failed */
            return DS_E_NO_MEMORY;
        }

        if ((n_byte_alloc /= 2) < n_byte_alloc_min) { /* last try */
            n_byte_alloc = n_byte_alloc_min;
        }
    } /* end of loop trying smaller allocations */

    /* Copy to the new buffer */
    (void) memcpy(p_buf_new, p_ds->p_buf, p_ds->length + 1);

    /* If there already was a dynamic allocation, free it */
    if (p_ds->p_buf != p_ds->p_stack_buf) {
        txfree((void *) p_ds->p_buf);
    }

    /* Assign new active buffer and its size */
    p_ds->p_buf = p_buf_new;
    p_ds->n_byte_alloc = n_byte_alloc;

    return DS_E_OK;
} /* end of function ds_reserve_nocheck */



/* Concatenate the result of a printf-style format
 *
 * Return codes as for ds_cat_vprintf */
int ds_cat_printf(DSTRING *p_ds, const char *sz_fmt, ...)
{
    va_list p_arg;
    va_start(p_arg, sz_fmt);
    const int xrc = ds_cat_vprintf(p_ds, sz_fmt, p_arg);
    va_end(p_arg);
    return xrc;
} /* end of function ds_cat_printf */



/* Concatenate the result of a printf-style format using va_list
 *
 * Return codes
 * DS_E_OK: Formatted OK
 * DS_E_NO_MEMORY: Unable to allocate memory to resize buffer
 * DS_E_INVALID: Invalid formatter / data
 */
int ds_cat_vprintf(DSTRING *p_ds, const char *sz_fmt, va_list p_arg)
{
    /* Make a copy of the argument list in case need to format more than
     * once */
    va_list p_arg2;
    va_copy(p_arg2, p_arg);
    const size_t n_byte_free = p_ds->n_byte_alloc - p_ds->length;
    char * const p_dst = p_ds->p_buf + p_ds->length;
    const int rc = vsnprintf(p_dst, n_byte_free, sz_fmt, p_arg);
    if (rc < 0) { /* Check for formatting error */
        return DS_E_INVALID;
    }

    /* Else check for buffer large enough and set length if it is */
    if ((size_t) rc < n_byte_free) {
        p_ds->length += (size_t) rc;
        return DS_E_OK;
    }

    /* Else buffer too small, so resize and format again */
    {
        /* Double required size to avoid excessive allocations +1 for
         * null, which is not included in the count returned by snprintf */
        const size_t n_byte_alloc_min =
                p_ds->length + (size_t) rc + (size_t) 1;
        if (ds_reserve_internal(p_ds,
                2 * n_byte_alloc_min, n_byte_alloc_min) == DS_E_NO_MEMORY) {
            /* vsnprintf may have written bytes to the buffer.
             * Ensure that dstring in a consistent state by writing
             * a null at the length of the string */
            p_ds->p_buf[p_ds->length] = '\0';
            return DS_E_NO_MEMORY;
        }
        const size_t n_byte_free2 = p_ds->n_byte_alloc - p_ds->length;
        char * const p_dst2 = p_ds->p_buf + p_ds->length;
        const int rc2 = vsnprintf(p_dst2, n_byte_free2, sz_fmt, p_arg2);
        if (rc2 < 0) { /* Check for formatting error */
            /* vsnprintf may have written bytes to the buffer.
             * Ensure that dstring in a consistent state by writing
             * a null at the length of the string */
            p_ds->p_buf[p_ds->length] = '\0';
            return DS_E_INVALID;
        }

        /* Else update length. No need to check buffer size since it was
         * sized to fit the string. */
        p_ds->length += (size_t) rc2;
        return DS_E_OK;
    }
} /* end of function ds_cat_vprintf */




/* Reallocate/free to eliminate unused buffer space.
 *
 * Return codes
 * DS_E_OK: Compacted OK
 * DS_E_NO_MEMORY: Compaction failed, but dstring still valid */
int ds_compact(DSTRING *p_ds)
{
    const size_t n_byte_alloc_min = p_ds->length + 1;

    /* If the string is in the stack buffer, there is nothing to do */
    if (p_ds->p_stack_buf == p_ds->p_buf) {
        return DS_E_OK;
    }

    /* Else if the string will fit in the stack buffer, copy it there and
     * free the allocation. */
    if (p_ds->n_byte_stack_buf >= n_byte_alloc_min) {
        (void) memcpy(p_ds->p_stack_buf, p_ds->p_buf, n_byte_alloc_min);
        txfree((void *) p_ds->p_buf);
        p_ds->p_buf = p_ds->p_stack_buf;
        p_ds->n_byte_alloc = p_ds->n_byte_stack_buf;
        return DS_E_OK;
    }

    /* Else if the heap buffer is the minimum size, there is nothng to do */
    if (n_byte_alloc_min == p_ds->n_byte_alloc) {
        return DS_E_OK;
    }

    /* Else realloc the heap buffer */
    {
        void *p = TREALLOC(char, p_ds->p_buf, n_byte_alloc_min);
        if (p == NULL) {
            return DS_E_NO_MEMORY;
        }
        p_ds->p_buf = (char *) p;
        p_ds->n_byte_alloc = n_byte_alloc_min;
        return DS_E_OK;
    }
} /* end of function ds_compact */



#ifdef DSTRING_UNIT_TEST
#if defined (_WIN32) && !defined(CONSOLE)
#include "ngspice/wstdio.h"
#endif
static void ds_print_info(DSTRING *p_ds, FILE *fp, const char *sz_id);
static int ds_test_from_macro(FILE *fp);
static int ds_test_from_stack(FILE *fp);
static int ds_test_from_heap(FILE *fp);
static int ds_test1(DSTRING *p_ds, FILE *fp);


int ds_test(FILE *fp)
{
    if (ds_test_from_macro(fp) != 0) { /* create from macro and run test */
        return -1;
    }
    if (ds_test_from_stack(fp) != 0) { /* create from stack */
        return -1;
    }
    if (ds_test_from_heap(fp) != 0) { /* create from heap */
        return -1;
    }

    return 0;
} /* end of function ds_test */



/* Run tests from a macro-created dstring */
static int ds_test_from_macro(FILE *fp)
{
    DS_CREATE(ds, 10);
    (void) fprintf(fp, "Macro initialization\n");
    return ds_test1(&ds, fp);
} /* end of function ds_test_from_macro */



/* Run tests from a manually created stack-backed dstring */
static int ds_test_from_stack(FILE *fp)
{
    static char p_buf[30] = "Hello World";
    DSTRING ds;
    (void) fprintf(fp, "Stack initialization\n");
    (void) ds_init(&ds, p_buf, 11, sizeof p_buf,  ds_buf_type_stack);
    return ds_test1(&ds, fp);
} /* end of function ds_test_from_stack */



/* Run tests from a heap-backed dstring */
static int ds_test_from_heap(FILE *fp)
{
    char *p_buf = (char *) malloc(25);
    if (p_buf == (char *) NULL) {
        return -1;
    }
    (void) memcpy(p_buf, "Heap", 4);
    DSTRING ds;
    (void) ds_init(&ds, p_buf, 4, 25,  ds_buf_type_heap);
    (void) fprintf(fp, "Heap initialization\n");
    return ds_test1(&ds, fp);
} /* end of function ds_test_from_heap */



static int ds_test1(DSTRING *p_ds, FILE *fp)
{
    /* Print info on entry */
    ds_print_info(p_ds, fp, "On entry to ds_test1\n");

    int i;
    for (i = 0; i < 10; i++) {
        if (ds_cat_str(p_ds, "Abc") != 0) {
            (void) fprintf(fp, "Unable to cat string %d.\n", i);
            return -1;
        }
        if (ds_cat_str_case(p_ds, "Abc", ds_case_as_is) != 0) {
            (void) fprintf(fp, "Unable to cat string as-is %d.\n", i);
            return -1;
        }
        if (ds_cat_str_case(p_ds, "Abc", ds_case_upper) != 0) {
            (void) fprintf(fp, "Unable to cat string upper %d.\n", i);
            return -1;
        }
        if (ds_cat_str_case(p_ds, "Abc", ds_case_lower) != 0) {
            (void) fprintf(fp, "Unable to cat string lower %d.\n", i);
            return -1;
        }
        if (ds_cat_char(p_ds, 'z') != 0) {
            (void) fprintf(fp, "Unable to cat char %d.\n", i);
            return -1;
        }
        if (ds_cat_char_case(p_ds, 'z', ds_case_as_is) != 0) {
            (void) fprintf(fp, "Unable to cat char as-is %d.\n", i);
            return -1;
        }
        if (ds_cat_char_case(p_ds, 'z', ds_case_upper) != 0) {
            (void) fprintf(fp, "Unable to cat char upper %d.\n", i);
            return -1;
        }
        if (ds_cat_char_case(p_ds, 'Z', ds_case_lower) != 0) {
            (void) fprintf(fp, "Unable to cat char lower %d.\n", i);
            return -1;
        }

        if (ds_cat_mem(p_ds, "Zyxw", 4) != 0) {
            (void) fprintf(fp, "Unable to cat string %d.\n", i);
            return -1;
        }
        if (ds_cat_mem_case(p_ds, "Zyxw", 4, ds_case_as_is) != 0) {
            (void) fprintf(fp, "Unable to cat string as-is %d.\n", i);
            return -1;
        }
        if (ds_cat_mem_case(p_ds, "Zyxw", 4, ds_case_upper) != 0) {
            (void) fprintf(fp, "Unable to cat string upper %d.\n", i);
            return -1;
        }
        if (ds_cat_mem_case(p_ds, "Zyxw", 4, ds_case_lower) != 0) {
            (void) fprintf(fp, "Unable to cat string lower %d.\n", i);
            return -1;
        }

        if (ds_cat_printf(p_ds, "--- And finally a formatted %s (%d)",
                "string", i) != 0) {
            (void) fprintf(fp, "Unable to cat formatted string %d.\n", i);
            return -1;
        }

        /* Print info after cats */
        ds_print_info(p_ds, fp, "After appending strings");

        /* Truncate the string */
        if (ds_set_length(p_ds, i * (size_t) 10) != 0) {
            (void) fprintf(fp, "Unable to set size %d.\n", i);
            return -1;
        }

        /* Print info after truncation */
        ds_print_info(p_ds, fp, "After setting length");

        /* Compact the string */
        if (ds_compact(p_ds) != 0) {
            (void) fprintf(fp, "Unable to compact %d.\n", i);
            return -1;
        }

        /* Print info after compaction */
        ds_print_info(p_ds, fp, "After compacting the string");
    } /* end of loop over tests */

    ds_free(p_ds); /* free buffer if allocated */

    return 0;
} /* end of funtion ds_test */



/* Print some info about the DSTRING */
static void ds_print_info(DSTRING *p_ds, FILE *fp, const char *sz_id)
{
    (void) fprintf(fp, "%s: length = %zu; "
            "allocated buffer size = %zu; value = \"%s\"; "
            "address of active buffer = %p; "
            "address of stack buffer = %p; "
            "size of stack buffer = %zu\n",
            sz_id,
            ds_get_length(p_ds), ds_get_buf_size(p_ds),
            ds_get_buf(p_ds), ds_get_buf(p_ds),
            p_ds->p_stack_buf, p_ds->n_byte_stack_buf);
} /* end of function ds_print_info */



#endif /* DSTRING_UNIT_TEST */



