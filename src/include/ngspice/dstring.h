/*   dstring.h    */

#ifndef DSTRING_H
#define DSTRING_H


#include <stdarg.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include "ngspice/memory.h"


/* Error codes */
#define DS_E_OK         0
#define DS_E_INVALID    (-1)
#define DS_E_NO_MEMORY  (-2)

/* Macros to create and initialize the most common type of dstring, which is
 * one that uses the stack for the initial buffer and begins empty.
 *
 * Example:
 *
 * DS_CREATE(ds1, 50); // Creates dstring ds1 backed by 50 bytes of stack
 *                     // memory and initialized to "".
 * Note that each DS_CREATE macro must be on a separate line due to the use
 * of the __LINE__ macro. Using __COUNTER__ in its place would resolve this
 * issue, but __COUNTER__ is not part of the ANSI standard.
 */
#undef DS_CONCAT
#undef DS_CONCAT2
#define DS_CONCAT2(a, b) a##b
#define DS_CONCAT(a, b) DS_CONCAT2(a, b)
#define DS_CREATE(ds_name, n) \
    char DS_CONCAT(ds_buf___, __LINE__)[n]; \
    DSTRING ds_name; \
    ds_init(&ds_name, DS_CONCAT(ds_buf___, __LINE__), 0,\
            sizeof DS_CONCAT(ds_buf___, __LINE__), ds_buf_type_stack)


/* Structure for maintaining a dynamic string */
typedef struct Dstring {
  char *p_buf; /* Active data buffer */
  size_t length; /* Number of characters in the string excluding the
                  * terminating NULL. */
  size_t n_byte_alloc; /* Allocated size of current  buffer */
  char *p_stack_buf; /* address of stack-based buffer backing dstring
                      * or NULL if none */
  size_t n_byte_stack_buf; /* size of stack_buffer or 0 if none */
} DSTRING, *DSTRINGPTR;


/* Enumeration defining buffer types used during initialization */
typedef enum ds_buf_type {
    ds_buf_type_stack, /* Buffer allocated from stack */
    ds_buf_type_heap /* Buffer allocated from heap */
} ds_buf_type_t;

/* Enumeration defining case conversion */
typedef enum ds_case {
    ds_case_as_is, /* Leave characters as they are */
    ds_case_lower, /* Losercase chars */
    ds_case_upper /* Uppercase chars */
} ds_case_t;



/* General initialization */
int ds_init(DSTRING *p_ds, char *p_buf, size_t length_string,
        size_t n_byte_buf, ds_buf_type_t type_buffer);

/* Free all memory used */
void ds_free(DSTRING *p_ds);


/* Concatenate string */
int ds_cat_str_case(DSTRING *p_ds, const char *sz, ds_case_t case_type);
inline int ds_cat_str(DSTRING *p_ds, const char *sz)
{
    return ds_cat_str_case(p_ds, sz, ds_case_as_is);
} /* end of function ds_cat_str */



/* Concatenate character */
int ds_cat_char_case(DSTRING *p_ds, char c, ds_case_t case_type);
inline int ds_cat_char(DSTRING *p_ds, char c)
{
    return ds_cat_char_case(p_ds, c, ds_case_as_is);
} /* end of function ds_cat_char */



/* Concatenate another dstring */
int ds_cat_ds_case(DSTRING *p_ds_dst, const DSTRING *p_ds_src,
        ds_case_t case_type);
inline int ds_cat_ds(DSTRING *p_ds_dst, const DSTRING *p_ds_src)
{
    return ds_cat_ds_case(p_ds_dst, p_ds_src, ds_case_as_is);
} /* end of function ds_cat_ds */



/* General concatenation of a memory buffer */
int ds_cat_mem_case(DSTRING *p_ds, const char *p_src, size_t n_char,
        ds_case_t type_case);
inline int ds_cat_mem(DSTRING *p_ds, const char *p_src, size_t n_char)
{
    return ds_cat_mem_case(p_ds, p_src, n_char, ds_case_as_is);
} /* end of function ds_cat_mem */



/* Ensure minimum internal buffer size */
int ds_reserve(DSTRING *p_ds, size_t n_byte_alloc_min);

/* Concatenate the result of a printf-style format */
int ds_cat_printf(DSTRING *p_ds, const char *sz_fmt, ...);

/* Concatenate the result of a printf-style format using va_list */
int ds_cat_vprintf(DSTRING *p_ds, const char *sz_fmt, va_list p_arg);

/* Reallocate/free to eliminate unused buffer space */
int ds_compact(DSTRING *p_ds);



/* This function sets the length of the buffer to some size less than
 * the current allocated size
 *
 * Return codes
 * DS_E_OK: length set OK
 * DS_E_INVALID: length to large for current allocation
 */
inline int ds_set_length(DSTRING *p_ds, size_t length)
{
    if (length >= p_ds->n_byte_alloc) {
        return DS_E_INVALID;
    }
    p_ds->length = length;
    p_ds->p_buf[p_ds->length] = '\0';
    return DS_E_OK;
} /* end of function ds_set_length */



/* Sets the length of the data in the buffer to 0. It is equivalent to
 * ds_set_length(p_ds, 0), except that the check for a valid length can
 * be skipped since 0 is always valid. */
inline void ds_clear(DSTRING *p_ds)
{
    p_ds->length = 0;
    p_ds->p_buf[0] = '\0';
} /* end of function ds_clear */



/* This function, if successful, returns an allocated buffer with the
 * string to the caller and frees any other resources used by the DSTRING.
 * If the buffer is not allocated and the DS_FREE_MOVE_OPT_FORCE_ALLOC
 * option is not selected, NULL is returned.
 *
 * Parameters
 * p_ds: Address of DSTRING to free
 * opt: Bitwise options
 *      DS_FREE_MOVE_OPT_FORCE_ALLOC -- Force allocation in all cases.
 *          If fails, the DSTRING is unchanged.
 *      DS_FREE_MOVE_OPT_COMPACT -- Resize allocation to minimum size
 *          if one already exists.
 *
 * Return values
 * The data string is returned as an allocation to be freed by the caller
 * NULL is returned if either the allocation was stack-based and
 *      DS_FREE_MOVE_OPT_FORCE_ALLOC was not selected or if
 *      DS_FREE_MOVE_OPT_COMPACT or DS_FREE_MOVE_OPT_FORCE_ALLOC
 *      options were given and there was an allocation failure.
 *      In any case when NULL is returned, the DSTRING is unchanged
 *      on return.
 *
 * Remarks
 * To force freeing of resources if this function fails, either it can
 * be called again with no options or equivalently ds_free() can be used.
 */
#define DS_FREE_MOVE_OPT_FORCE_ALLOC    1
#define DS_FREE_MOVE_OPT_COMPACT        2
inline char *ds_free_move(DSTRING *p_ds, unsigned int opt)
{
    char * const p_buf_active = p_ds->p_buf;

    /* If the buffer is from the stack, allocate if requested. Note that the
     * compaction option is meaningless in this case since it is allocated
     * to the minimum size required */
    if (p_buf_active == p_ds->p_stack_buf) { /* not allocated */
        if (opt & DS_FREE_MOVE_OPT_FORCE_ALLOC) {
            /* Allocate to minimum size */
            size_t n_byte_alloc = p_ds->length + 1;
            char * const p_ret = TMALLOC(char, n_byte_alloc);
            if (p_ret == (char *) NULL) {
                return (char *) NULL;
            }
            return memcpy(p_ret, p_buf_active, n_byte_alloc);
        }
        return (char *) NULL;
    }
    /* Else allocated */
    if (opt & DS_FREE_MOVE_OPT_COMPACT) {
        /* Allocate to minimum size */
        size_t n_byte_alloc = p_ds->length + 1;
        char * const p_ret = TREALLOC(char, p_buf_active, n_byte_alloc);
        if (p_ret == (char *) NULL) {
            /* Realloc to smaller size somehow failed! */
            return (char *) NULL;
        }
        return p_ret; /* Return resized allocation */
    }
    return p_buf_active; /* Return unchanged */
} /* end of function ds_free_move */



/* Returns the address of the buffer. The caller should never attempt
 * to free the buffer. With care (not changing the length), it can
 * be modified. */
inline char *ds_get_buf(DSTRING *p_ds)
{
    return p_ds->p_buf;
} /* end of function ds_get_buffer */



/* Returns the current dstring length */
inline size_t ds_get_length(const DSTRING *p_ds)
{
    return p_ds->length;
} /* end of function ds_get_length */



/* Returns the allocated dstring buffer size */
inline size_t ds_get_buf_size(const DSTRING *p_ds)
{
    return p_ds->n_byte_alloc;
} /* end of function ds_get_buf_size */



#ifdef DSTRING_UNIT_TEST
#include <stdio.h>
int ds_test(FILE *fp);
#endif /* UNIT_TEST_DSTRING */

#endif /* include guard */
