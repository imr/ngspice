/* Functions for managing history of strings, such as commands
 *
 * Implemented using circular buffers for both the storage of the
 * strings and their locating information.
 */

#include <limits.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include "hist_info.h"


/* Structure locating string informaton */
struct Str_info {
    unsigned int n_byte_sz; /* length of string with NULL at end */
    char *sz; /* Address of string */
};
struct History_info {
    struct History_info_opt hi_opt;
    bool f_first_resize_check_done;
    unsigned int n_str_cur; /* current number of strings */
    unsigned int n_str_alloc; /* allocated size of array */
    unsigned int n_insert_since_resize_check;
                        /* For buffer size management. After this many more
                         * string insertions, the buffer will be reduced
                         * in size if it is excessively large for the data */
    size_t n_byte_buf_cur; /* Current amount of history buffer in use */
    size_t n_byte_buf_alloc; /* Allocated size of history buffer */
    unsigned int index_str_start; /* Index of first history string item */
    unsigned int index_str_cur; /* Index of next string */
    unsigned int index_str_to_return; /* Index of string to return */
    char *p_char_buf; /* Address of char buffer for history strings */
    char *p_char_buf_start; /* Start of data in buffer */
    char *p_char_buf_cur; /* Current free address in buffer */
    char *p_char_buf_end; /* Byte past last address in buffer */

    /* Array of n_max items locating commnads. This is a circular buffer
     * with the data elements being from index_start to index_end-1,
     * inclusive of the endpoints with wrapping considered */
    struct Str_info p_str_info[1];
    
    /* Buffer for string history strings follows p_str_info array */
}; /* end of struct History_info */


static int adjust_history_options(struct History_info_opt *p_hi_opt,
        bool f_is_init);
static struct History_info *history_alloc(
        unsigned int n_str_max, size_t n_byte_char_buf);
static int history_copy(struct History_info *p_hi_dst,
        const struct History_info *p_hi_src);
static const char *history_get_prev1(struct History_info *p_hi,
        unsigned int *p_n_char_str, bool f_update_pos);
static void history_make_empty(struct History_info *p_hi);
static int history_resize(struct History_info **pp_hi,
        unsigned int n_str_alloc, size_t n_byte_char_buf_alloc);
static inline const char *return_str_data(struct History_info *p_hi,
        unsigned int index_str_to_return, unsigned int *p_n_char_str);

/* This function allocates and initializes history information
 *
 * Parameter
 * n_max: Maximum number of history items to store
 *
 * Return values
 * NULL on error; otherwise an initialized history structure. Options
 * are modified to actual values used. */
struct History_info *history_init(struct History_info_opt *p_hi_opt)
{
    struct History_info *p_hi;

    /* Make history options valid */
    if (adjust_history_options(p_hi_opt, true) < 0) {
        return (struct History_info *) NULL;
    }

    /* Do allocation */
    if ((p_hi = history_alloc(p_hi_opt->n_str_init,
            p_hi_opt->n_byte_str_buf_init)) ==
            (struct History_info *) NULL) {
        return (struct History_info *) NULL;
    }

    /* Save options */
    p_hi->hi_opt = *p_hi_opt;

    p_hi->f_first_resize_check_done = false; 

    return p_hi;
} /* end of function history_init */



/* This function modifies history options to valid  values
 *
 * Return codes
 * +1: Modified OK
 * 0: No changes required
 * -1: Unknown option structure size */
static int adjust_history_options(struct History_info_opt *p_hi_opt,
        bool f_is_init)
{
    /* Validate and adjust arguments */
    if (p_hi_opt->n_byte_struct != sizeof(struct History_info_opt)) {
        /* Unknown version */
        return -1;
    }

    int xrc = 0;

    /* Must be at least 2 strings buffered */
    if (p_hi_opt->n_str_init < 2) {
        if (f_is_init) {
            xrc = +1;
        }
        p_hi_opt->n_str_init = 2;
    }

    /* If initialization, max # strings buffered must be at least init
     * size */
    if (f_is_init) {
        if (p_hi_opt->n_str_max < p_hi_opt->n_str_init) {
            xrc = +1;
            p_hi_opt->n_str_max = p_hi_opt->n_str_init;
        }
    }

    /* Initial string buffer must be at least 2 bytes */

    if (p_hi_opt->n_byte_str_buf_init < 2) {
        if (f_is_init) {
            xrc = +1;
        }
        p_hi_opt->n_byte_str_buf_init = 2;
    }

    /* Oversize factor must be at least 4 */
    if (p_hi_opt->oversize_factor < 4) {
        xrc = +1;
        p_hi_opt->oversize_factor = 4;
    }

    return xrc;
} /* end of function adjust_history_options */



/* This function allocates history information
 *
 * Parameters
 * n_max: Maximum number of history items to store
 * n_byte_buf: Character buffer size in bytes
 *
 * Return values
 * NULL on allocation failure;
 * otherwise an initialized history structure */
static struct History_info *history_alloc(
        unsigned int n_str, size_t n_byte_char_buf)
{
    struct History_info *p_hi;
    
    /* Memory offset to history buffer from start of structure
     * Note that no alignment is required for char */
    const size_t offset_char_buf = sizeof(struct History_info) +
            sizeof(struct Str_info) * (n_str - 1);

    /* Total allocation size */
    const size_t n_byte_alloc = offset_char_buf + n_byte_char_buf;

    /* Allocate history buffer */
    if ((p_hi = (struct History_info *) malloc(n_byte_alloc)) ==
            (struct History_info *) NULL) {
        return (struct History_info *) NULL;
    }

    p_hi->n_str_alloc = n_str;
    p_hi->n_insert_since_resize_check = 0;
    p_hi->n_byte_buf_alloc = n_byte_char_buf;

    {
        /* Locate start and end of string buffer */
        char *p_cur = (char *) p_hi + offset_char_buf;
        p_hi->p_char_buf = p_cur;
        p_cur += n_byte_char_buf;
        p_hi->p_char_buf_end = p_cur;
    }

    /* Initialize the buffer and locator array to empty. */
    history_make_empty(p_hi);

    return p_hi;
} /* end of function history_alloc */



/* This function copies one history state into another. It is useful when
 * a history buffer needs to be resized or the number of items in the
 * commnad history is changed. The source history info must have at least
 * one string.
 *
 * Parameters
 * p_hi_dst: Address of destination history info
 * p_hi_src: Address of source history info
 *
 * Return codes
 * +1: Destination buffer is too small. Data was truncated.
 * 0: Data copied OK.
 *
 *  Note: if there are more history strings in the source structure
 *  than the n_str_max field in the destinnation, only the newest strings
 *  are copied.
 */
static int history_copy(struct History_info *p_hi_dst,
        const struct History_info *p_hi_src)
{
    int xrc = 0;

    const unsigned int n_str_src = p_hi_src->n_str_cur; /* # src strs avail */

    /* Copy data directly taken from source */
    p_hi_dst->hi_opt = p_hi_src->hi_opt; /* Copy options */
    p_hi_dst->f_first_resize_check_done =
            p_hi_src->f_first_resize_check_done;
    p_hi_dst->n_insert_since_resize_check =
            p_hi_src->n_insert_since_resize_check;

    /* If the source is empty, set dest to empty */
    if (n_str_src == 0) {
        history_make_empty(p_hi_dst);
        return 0;
    }

    const unsigned int n_str_src_alloc  = p_hi_src->n_str_alloc; /* size */
    unsigned int index_cur_src; /* current index to copy */
    unsigned int index_str_start_src; /* index of oldest item in source
                                       * to be copied */

    /* Find the first source index to copy using index_cur_src as a temp
     * variable */
    if (n_str_src > p_hi_dst->n_str_alloc) {
        /* If there are more source strings than the number of available
         * entries in the destination, only copy the newest items. May have
         * to handle a wrap to the start of the buffer. */
        if ((index_str_start_src = p_hi_src->index_str_start +
                (n_str_src - p_hi_dst->n_str_alloc)) >= n_str_src_alloc) {
            index_str_start_src -= n_str_src_alloc;
        }
        xrc = +1; /* indicate truncation */
    }
    else {
        /* Enough elements in destination for all strings, so start at the
         * beginning */
        index_str_start_src = p_hi_src->index_str_start;
    }

    /* Now make index_cur_src equal to the newest source history item */
    if ((index_cur_src = p_hi_src->index_str_cur - 1) == (unsigned int) -1) {
        /* Underflow occurred, so overflow to correct */
        index_cur_src += p_hi_src->n_str_alloc; /* Set to end */
    }

    /* Start adding strings at bottom of Str_info buffer */
    unsigned int index_cur_dst = p_hi_dst->n_str_alloc - 1;

    /* Start writing strings at botttom of char buffer. This location is
     * actually the address of the top of the buffer unrolled to the byte
     * past its end. */
    char *p_dst_cur = p_hi_dst->p_char_buf_end;
    const char *p_char_buf_dst = p_hi_dst->p_char_buf;

    size_t n_byte_buf_cur_dst = 0; /* bytes copied to destination */
    const struct Str_info * const p_str_info_src0 = p_hi_src->p_str_info;
    struct Str_info * const p_str_info_dst0 = p_hi_dst->p_str_info;

    /* Copy each element in the source string history starting with the
     * newest element so that if truncation occurs, it will be the oldest
     * strings that get truncated */
    for ( ; ; ) {
        /* Locate data to copy */
        const struct Str_info * const p_str_info_src =
                p_str_info_src0 + index_cur_src;
        const unsigned int n_byte_str_cur = p_str_info_src->n_byte_sz;
        struct Str_info * const p_str_info_dst = p_str_info_dst0 +
                index_cur_dst;

        /* Locate the destination in the char * buffer */
        if ((p_dst_cur -= n_byte_str_cur) < p_char_buf_dst) {
            break; /* Data will not fit */
        }

        /* Copy the data */
        (void) memcpy(p_dst_cur, p_str_info_src->sz, n_byte_str_cur);
        n_byte_buf_cur_dst += n_byte_str_cur;

        /* Save locating information */
        p_str_info_dst->sz = p_dst_cur;
        p_str_info_dst->n_byte_sz = n_byte_str_cur;

        /* Test for exit, which is completion of copy of string at
         * index_str_src. On exit, index_cur_dst is at last string
         * copied (which is the oldest string copied) */
        if (index_cur_src == index_str_start_src) {
            break;
        }

        /* Move to next indices */
        if (--index_cur_src == (unsigned int) -1) {
            index_cur_src += n_str_src_alloc;
        }
        --index_cur_dst; /* by construction will not wrap */
    } /* end of loop copying strings */
        

    /* Complete the information for the copied history info */
    p_hi_dst->n_str_cur = p_hi_dst->n_str_alloc - index_cur_dst;
    p_hi_dst->n_byte_buf_cur = n_byte_buf_cur_dst;
    p_hi_dst->index_str_start = index_cur_dst; /* last copied is oldest */
    p_hi_dst->index_str_cur = 0; /* next item wraps to begin at top of buf */
    p_hi_dst->p_char_buf_start = p_dst_cur; /* oldest string (last copied) */
    p_hi_dst->p_char_buf_cur = p_hi_dst->p_char_buf;
                                /* top of char buf is next free byte */

    /* Set the index to return for history_get_next and history_get_prev.
     * If the earlier command no longer exists, set to the closest value */
    {
        /* Look in the source history information to see the location
         * relative to the current insert position */
        const unsigned int index_str_to_return =
                p_hi_src->index_str_to_return;
        if (index_str_to_return == UINT_MAX) {
            p_hi_dst->index_str_to_return = UINT_MAX;
        }
        else {
            const unsigned index_str_start = p_hi_src->index_str_start;
            const unsigned int index_str_cur = p_hi_src->index_str_cur;

            /* "Unrolled" positions -- as if wrapped part extended past
             * end of buffer */
            const unsigned index_str_cur_unrolled =
                    index_str_cur > index_str_start ? index_str_cur :
                        index_str_cur + p_hi_src->n_str_alloc;
            const unsigned int index_str_to_return_unrolled =
                    index_str_to_return > index_str_start ?
                            index_str_to_return :
                            index_str_to_return + p_hi_src->n_str_alloc;

            /* From unrolled positions, offset is always a simple
             * subtraction. With the offset being from the position to
             * return, the offset is always nonnegative */
            const unsigned int offset_from_cur = index_str_cur_unrolled -
                    index_str_to_return_unrolled;

            /* The offset must be reduced if the command being indexed was
             * not copied due to truncation */
            const unsigned int offset_from_cur_dst =
                    offset_from_cur <= p_hi_dst->n_str_cur ?
                            offset_from_cur :
                            p_hi_dst->n_str_cur;

            /* Locate the index for the string to return */
            if (p_hi_dst->index_str_cur >= offset_from_cur_dst) {
                /* No buffer wrap */
                p_hi_dst->index_str_to_return =
                        p_hi_dst->index_str_cur - offset_from_cur_dst;
            }
            else { /* Wrap */
                p_hi_dst->index_str_to_return =
                        p_hi_dst->index_str_cur +
                        (p_hi_dst->n_str_alloc - offset_from_cur_dst);
            }
        }
    } /* end of block locating index for returned history item */

    return xrc;
} /* end of function history_copy */



/* Set values to make an empty history buffer */
static void history_make_empty(struct History_info *p_hi)
{ 
    p_hi->n_str_cur = 0;
    p_hi->n_byte_buf_cur = 0;
    p_hi->index_str_start = 0;
    p_hi->index_str_cur = 0;
    p_hi->index_str_to_return = UINT_MAX;
    p_hi->p_char_buf_start = p_hi->p_char_buf_cur = p_hi->p_char_buf;
} /* end of function history_copy_empty */



/* This function returns the previous history item
 * For the previous item, the value returned should be one step
 * behind. That is, the first "previous" string is the current
 * string. It is done because there is a new current string not yet
 * in the history buffer when the previous one is being requested.
 *
 * Cases
 * Empty buffer -- n_str_cur == 0
 *      Return empty string
 * Buffer not empty
 *      Decrement index_str_to_return. If < index_str_start,
 *      set to index_str_cur - 1
 */
const char *history_get_prev(struct History_info *p_hi,
        unsigned int *p_n_char_str)
{
    return history_get_prev1(p_hi, p_n_char_str, true);
} /* end of function history_get_prev */



/* Worker function for history_get_prev and history_get_last that provides
 * the option to not change the current position */
static const char *history_get_prev1(struct History_info *p_hi,
        unsigned int *p_n_char_str, bool f_update_pos)
{
    const unsigned int n_str = p_hi->n_str_cur;

    /* Handle caase of empty history info */
    if (n_str == 0) {
        /* Set size if buffer given */
        if (p_n_char_str != (unsigned *) NULL) {
            *p_n_char_str = 0;
        }
        return "";
    }

    unsigned int index_str_to_return = f_update_pos &&
            p_hi->index_str_to_return != UINT_MAX ?
            p_hi->index_str_to_return : p_hi->index_str_cur;

    if (n_str == p_hi->n_str_alloc) { /* Full buffer */
        if (index_str_to_return == 0) { /* must wrap */
            index_str_to_return = n_str - 1;
        }
        else { /* wrap not required */
            --index_str_to_return;
        }
    }
    else { /* Partial buffer */
        if (index_str_to_return == 0) { /* must wrap */
            if (p_hi->index_str_start < p_hi->index_str_cur) {
                /* Data in [start, end-1] */
                index_str_to_return = p_hi->index_str_cur - 1;
            }
            else { /* end less than start (if ==, buffer is full) */
                /* Data in [start, n_str_max-1]. If end != 0, there
                 * is a second piece of data in [0, end-1] */
                index_str_to_return = p_hi->n_str_alloc - 1;
            }
        } /* end of case of wrap at 0 */
        else { /* current index_str_to_return > 0 */
            if (index_str_to_return == p_hi->index_str_start) {
                if (p_hi->index_str_cur == 0) {
                    /* last str at bottom of buf */
                    index_str_to_return = p_hi->n_str_alloc - 1;
                }
                else {
                    index_str_to_return = p_hi->index_str_cur - 1;
                }
            }
            else { /* no special cases, so just decrement */
                --index_str_to_return;
            }
        }
    }

    /* Save updated position */
    if (f_update_pos) {
        p_hi->index_str_to_return = index_str_to_return;
    }

    /* Return the string data */
    return return_str_data(p_hi, index_str_to_return, p_n_char_str);
} /* end of function history_get_prev */



/* This function returns the newest history item, that is the last one
 * added. It can be used to decide whether or not to add a string. For
 * example, duplicate consecutive strings can be suppressed. */
const char *history_get_newest(struct History_info *p_hi,
        unsigned int *p_n_char_str)
{
    return history_get_prev1(p_hi, p_n_char_str, false);
} /* end of function history_get_cur */



/* This function returns the next history item */
const char *history_get_next(struct History_info *p_hi,
        unsigned int *p_n_char_str)
{
    const unsigned int n_str = p_hi->n_str_cur;

    /* Handle caase of empty history info */
    if (n_str == 0) {
        /* Set size if buffer given */
        if (p_n_char_str != (unsigned *) NULL) {
            *p_n_char_str = 0;
        }
        return "";
    }


    unsigned int index_str_to_return = p_hi->index_str_to_return;
    if (index_str_to_return == UINT_MAX) {
        /* next item requested before any prevous ones were */
        index_str_to_return = p_hi->index_str_start;
    }
    else {
        if (n_str == p_hi->n_str_alloc) { /* Full buffer */
            if (index_str_to_return == n_str - 1) { /* must wrap */
                index_str_to_return = 0;
            }
            else { /* wrap not required */
                ++index_str_to_return;
            }
        }
        else { /* Partial buffer */
            if (index_str_to_return == p_hi->n_str_alloc - 1) {
                /* Must wrap */
                if (p_hi->index_str_start < p_hi->index_str_cur) {
                    /* Data in [start, end-1] */
                    index_str_to_return = p_hi->index_str_start;
                }
                else { /* end less than start (if ==, buffer is full) */
                    /* Data in [start, n_str_max-1]. If end != 0, there
                     * is a second piece of data in [0, end-1] */
                    if (p_hi->index_str_cur == 0) {
                        index_str_to_return = p_hi->index_str_start;
                    }
                    else {
                        index_str_to_return = 0;
                    }
                }
            } /* end of case of wrap at 0 */
            else { /* current index_str_to_return < max buf index */
                if (index_str_to_return == p_hi->index_str_cur - 1) {
                    /* not at end */
                    index_str_to_return = p_hi->index_str_start;
                }
                else { /* no special cases, so just increment */
                    ++index_str_to_return;
                }
            }
        }
    }

    /* Save updated position */
    p_hi->index_str_to_return = index_str_to_return;

    /* Return the string data */
    return return_str_data(p_hi, index_str_to_return, p_n_char_str);
} /* end of function history_get_next */



/* This function returns the history information according to the given
 * index */
static inline const char *return_str_data(struct History_info *p_hi,
        unsigned int index_str_to_return, unsigned int *p_n_char_str)
{
    struct Str_info *p_str_info_cur = p_hi->p_str_info + index_str_to_return;
    /* Return the string. Also return size if a buffer was given */
    if (p_n_char_str != (unsigned *) NULL) {
        /* -1 because value stored includes NULL at end */
        *p_n_char_str = p_str_info_cur->n_byte_sz - 1;
    }
    
    return p_str_info_cur->sz;
} /* end of function return_str_data */



/* This function copies one history state into another. It is useful when
 * a history buffer needs to be resized or the number of items in the
 * commnad history is changed. The source history info must have at least
 * one string.
 *
 * Parameters
 * p_hi_dst: Address of destination history info
 * p_hi_src: Address of source history info
 *
 * Return codes
 * +1: Destination buffer is too small. Data was truncated.
 * 0: Buffer resized OK
 * -1: Allocation failure. Buffer same as when input
 *
 *  Note: if there are more history strings in the source structure
 *  than the n_str_max field in the destinnation, only the newest strings
 *  are copied.
 */
static int history_resize(struct History_info **pp_hi,
        unsigned int n_str_alloc, size_t n_byte_char_buf_alloc)
{
    struct History_info *p_hi_old = *pp_hi;
    struct History_info *p_hi_new;

    /* Allocate a new history info of the desired sizes */
    if ((p_hi_new = history_alloc(n_str_alloc, n_byte_char_buf_alloc)) ==
            (struct History_info *) NULL) {
        return -1;
    }

    /* Copy the old history into the new one */
    const int xrc = history_copy(p_hi_new, p_hi_old);

    /* Free the old allocation */
    history_free(p_hi_old);

    *pp_hi = p_hi_new; /* return new info */

    return xrc;
} /* end of function history_resize */



/* This function frees menory used by a History_info struct */
void history_free(struct History_info *p_hi)
{
    if (p_hi != (struct History_info *) NULL) {
        free((void *) p_hi);
    }

    return;
} /* end of function history_free */



/* This function adds the string str, of length n_char_str, excluding any
 * terminating null, which is optional. The history info always adds a
 * terminating null to the stored data.
 *
 * Return codes
 * 0: Added OK
 * -1: Unable to add.
 *
 * Remarks
 * The History_info structure may be allocated again to obtain more buffer
 * space. Failure of this allocation would result in a -1 return code.
 */
int history_add(struct History_info **pp_hi,
        unsigned int n_char_str, const char *str)
{
    const unsigned int n_byte_data = n_char_str + 1; /* with NULL */
    struct History_info *p_hi = *pp_hi; /* access history data */
    char *p_dst = (char *) NULL; /* Location to write new data */
    bool f_have_room; /* flag that there is room for the string */

    /* If the buffer is full of strings, resize, doubling up to the maximum
     * allowed size, and if that is not large enough, remove the oldest
     * one to make room for this one */
    if (p_hi->n_str_cur == p_hi->n_str_alloc) {
        unsigned int n_str_alloc_new = 2 * p_hi->n_str_alloc;
        if (n_str_alloc_new > p_hi->hi_opt.n_str_max) {
            n_str_alloc_new = p_hi->hi_opt.n_str_max;
        }

        /* If the buffer can be made larger, try to do so */
        if (n_str_alloc_new > p_hi->n_str_alloc) {
            if (history_resize(&p_hi, n_str_alloc_new,
                    p_hi->n_byte_buf_alloc) != 0) {
                return -1;
            }
            *pp_hi = p_hi; /* point to new structure */
            f_have_room = true;
        }
        else { /* at max size already */
            f_have_room = false;
        }
    }
    else { /* Allocated size not full yet */
        f_have_room = true;
    }


    /* If there is not room for the string, remove refrerence to the
     * oldest one to make room */
    if (!f_have_room) {
        p_hi->n_byte_buf_cur -=
                p_hi->p_str_info[p_hi->index_str_start].n_byte_sz;
        if (p_hi->index_str_start == p_hi->n_str_alloc - 1) {
            p_hi->index_str_start = 0;
        }
        else {
            ++p_hi->index_str_start;
        }

        /* Locate start of used buffer at the new starting string */
        p_hi->p_char_buf_start =
                p_hi->p_str_info[p_hi->index_str_start].sz;
        --p_hi->n_str_cur;
    }


    /* Try fitting the string in the free area. If that fails, the character
     * buffer will be enlarged via a new history information structure. */
    {
        /* Identify free area of buffer
         *      (1) [cur, end) + [top, start)
         *      (2) [cur, start)
         *      (3) none
         */
        ptrdiff_t case_id = p_hi->p_char_buf_cur - p_hi->p_char_buf_start;
        if (case_id > 0 || case_id == 0 && p_hi->n_str_cur == 0) {
            /* Case 1, including an empty buffer.
             * Try fitting the string from the free address to the end of the
             * the buffer. If that fails, try fitting from the start of the
             * buffer to the start of data, exclusive of start of data. */
            if (p_hi->p_char_buf_cur + n_byte_data <= p_hi->p_char_buf_end) {
                p_dst = p_hi->p_char_buf_cur;
            }
            else { /* whould not fit at end so try at top */
                if (p_hi->p_char_buf + n_byte_data <= p_hi->p_char_buf_start) {
                    p_dst = p_hi->p_char_buf;
                }
            }
        }
        else if (case_id < 0) { /* Case 2 */
            /* Try fitting the string from the free address to the start
             * of data, exclusive of start of data. */
            if (p_hi->p_char_buf_cur + n_byte_data <= p_hi->p_char_buf_start) {
                p_dst = p_hi->p_char_buf_cur;
            }
        }
        /* Else case 3: buffer is full, so cannot fit */
    }

    /* If the string would not fit, enlarge the char buffer and add */
    if (p_dst == (char *) NULL) {
        if (history_resize(&p_hi, p_hi->n_str_alloc,
                2 * p_hi->n_byte_buf_alloc + n_byte_data) != 0) {
            return -1; /* could not resize */
        }
        *pp_hi = p_hi; /* Update returned structure */

        /* Buffer was enlarged enough to guarantee fitting, so can add
         * without any checks */
        p_dst = p_hi->p_char_buf_cur;
    } /* end of case of resize */

    /* Add an entry into Str_info for this string */
    {
        struct Str_info *p_str_info = p_hi->p_str_info + p_hi->index_str_cur;
        p_str_info->sz = p_dst;
        p_str_info->n_byte_sz = n_byte_data;
    }

    /* Set index to next free entry in Str_info */
    if (++p_hi->index_str_cur == p_hi->n_str_alloc) {
        p_hi->index_str_cur = 0;
    }

    /* Flag the item to return as not set */
    p_hi->index_str_to_return = UINT_MAX;

    /* Update data size */
    p_hi->n_byte_buf_cur += n_byte_data;

    /* Save string data in char buffer at p_dst */
    (void) memcpy(p_dst, str, n_char_str);
    p_dst += n_char_str;
    *p_dst++ = '\0'; /* null terminate and point to byte after null */

    /* Update free location in char buffer, pointed to by p_dst after
     * writing the string */
    p_hi->p_char_buf_cur = p_dst;

    ++p_hi->n_str_cur; /* Increment the string count */

    /* Test if the buffer should be shrunk. Using >= instead of ==
     * due to possibility that history_setopt changed the threshold
     * to a smaller value */
    {
        bool f_do_check = false;
        unsigned int n_insert_since_check =
                ++p_hi->n_insert_since_resize_check;

        /* Test if need to check for oversize based on options */
        if (p_hi->f_first_resize_check_done) { /* after 1st */
            const unsigned int n =
                    p_hi->hi_opt.n_insert_per_oversize_check;
            /* Test for nonzero (supressing resizes) and threshold met */
            if (n != 0 && n_insert_since_check >= n) {
                f_do_check = true;
            }
        }
        else { /* first check */
            const unsigned int n =
                    p_hi->hi_opt.n_insert_first_oversize_check;
            /* Test for nonzero (supressing resizes) and threshold met */
            if (n != 0 && n_insert_since_check >= n) {
                f_do_check = true;
                p_hi->f_first_resize_check_done = true;
            }
        }

        /* Do check if enough inserts have occurred */
        if (f_do_check) {
            size_t n_byte_buf_alloc = p_hi->n_byte_buf_alloc;
                p_hi->n_insert_since_resize_check = 0;
            if (n_byte_buf_alloc > 4 &&
                    p_hi->n_byte_buf_cur * p_hi->hi_opt.oversize_factor <
                            n_byte_buf_alloc) {
                /* Buffer too large for existing data, so shrink it.
                 * Should that fail for some reason, simply continue with the
                 * existing buffer */
                (void) history_resize(&p_hi, p_hi->n_str_alloc,
                        n_byte_buf_alloc / 2);
                *pp_hi = p_hi; /* point to new structure */
            }
        }
    } /* end of block for oversized buffer actions */

    return 0;
} /* end of function history_add */



/* This function resets the returned buffer pointer so that the
 * last command is returned by history_get_prev() */
void history_reset_pos(struct History_info *p_hi)
{
    p_hi->index_str_to_return = UINT_MAX;
} /* end of function history_reset_pos */



/* This function sets history options after history has been initialized.
 * An initialized History_info_opt structure is passed. The values of
 * the structure will be modified to the closest allowable values if
 * the ones given cannot be used.
 *
 * The values for n_str_init and n_byte_str_buf_init are ignored.
 * The value for n_insert_first_oversize_check is also ignored if the
 * first check has already been performed.
 *
 * Return codes
 * +2: Unknown structure size. No action taken
 * +1: Option values were modified but changes were made OK
 * 0: Changes made OK
 * -1: Unable to make change(s)
 */
int history_setopt(struct History_info **pp_hi,
        struct History_info_opt *p_hi_opt)
{
    int rc_adj;

    /* Adjust options */
    if ((rc_adj = adjust_history_options(p_hi_opt, 0)) < 0) {
        /* Unknown version based on size of structure */
        return +2;
    }

    struct History_info *p_hi = *pp_hi; /* access struct */

    /* If the maximum number of items to log is being reduced below the
     * current allocated value, a copy of the buffer with the new array
     * size must be made. */
    {
        unsigned int n_str_max_new = p_hi_opt->n_str_max;
        if (p_hi->n_str_alloc > n_str_max_new) {
            if (history_resize(&p_hi, n_str_max_new,
                    p_hi->n_byte_buf_alloc) < 0) {
                return -1;
            }

            /* Override the maximum number of strings */
            p_hi->hi_opt.n_str_max = n_str_max_new;

            *pp_hi = p_hi; /* Point to new structure */
        }
    }

    /* Other option changes will work as the are processed */
    {
        struct History_info_opt *p_hi_opt_dst = &p_hi->hi_opt;
        p_hi_opt_dst->n_str_max = p_hi_opt->n_str_max;
        p_hi_opt_dst->oversize_factor = p_hi_opt->oversize_factor;
        p_hi_opt_dst->n_insert_first_oversize_check =
                p_hi_opt->n_insert_first_oversize_check;
        p_hi_opt_dst->n_insert_per_oversize_check =
                p_hi_opt->n_insert_per_oversize_check;
    }

    return rc_adj;
} /* end of function history_setopt */



/* This function gets current history options.
 *
 * Return codes
 * 0: Options returned OK
 * -1: Unknown structure size. Options not returned.
 */
int history_getopt(const struct History_info *p_hi,
        struct History_info_opt *p_hi_opt)
{
    /* Test for valid structure size */
    if (sizeof(struct History_info_opt) != p_hi_opt->n_byte_struct) {
        return -1;
    }

    *p_hi_opt = p_hi->hi_opt;
    return 0;
} /* end of function hstory_getopt */



