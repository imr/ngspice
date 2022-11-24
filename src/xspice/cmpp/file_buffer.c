#include <errno.h>
#include <limits.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "file_buffer.h"
#include "cmpp.h"

/* Default buffer size */
#define N_BYTE_FILEBUF_INIT_DFLT    16384


static int fb_fill(FILEBUF *p_fb);
static int fbget_quoted_unescaped_string(FILEBUF *p_fb,
        FBTYPE *p_type_found, FBOBJ *p_fbobj);
static size_t fb_make_space_at_end(FILEBUF *p_fb);
static int fbget_quoted_string(FILEBUF *p_fb,
        FBTYPE *p_type_found, FBOBJ *p_fbobj);
static int fbget_quoted_escaped_string(FILEBUF *p_fb,
        FBTYPE *p_type_found, FBOBJ *p_fbobj);
static int fbget_unquoted_string(FILEBUF *p_fb,
        unsigned int n_type_wanted, FBTYPE *p_type_wanted,
        FBTYPE *p_type_found, FBOBJ *p_fbobj);
static int fb_return_obj(FILEBUF *p_fb,
        unsigned int n_type_wanted, FBTYPE *p_type_wanted,
        FBTYPE *p_type_found, FBOBJ *p_fbobj);
static int fb_return_string(FILEBUF *p_fb,
        FBTYPE *p_type_found, FBOBJ *p_fbobj);
static int fb_skip_to_eol(FILEBUF *p_fb);
static int fb_skip_whitespace(FILEBUF *p_fb);


/* This function initializes a file-buffer read access to the named file.
 *
 * Parameters
 * filename: Name of file to be opened for reading
 * n_byte_buf_init: Intial buffer size. May be 0 for default size
 *
 * Return values
 * NULL: Error occurred. The value of errno will provide more details
 * Otherwise an initialized structure
 */
FILEBUF *fbopen(const char *filename, size_t n_byte_buf_init)
{
    int xrc = 0;
    FILEBUF *p_fb = (FILEBUF *) NULL;

    /* Set default buffer size if requested */
    if (n_byte_buf_init == 0) {
        n_byte_buf_init = N_BYTE_FILEBUF_INIT_DFLT;
    }

    /* Allocate structure to return */
    if ((p_fb = (FILEBUF *) malloc(sizeof *p_fb)) == (FILEBUF *) NULL) {
        xrc = -1;
        goto EXITPOINT;
    }

    p_fb->is_eof = false;
    p_fb->f_skip_to_eol = false;

    /* Init resources for error recovery */
    p_fb->fp = (FILE *) NULL;
    p_fb->p_buf = (char *) NULL;

    /* Allocate data buffer */
    if ((p_fb->p_buf = (char *) malloc(n_byte_buf_init)) == (char *) NULL) {
        xrc = -1;
        goto EXITPOINT;
    }
    p_fb->n_byte_buf_alloc = n_byte_buf_init;
    p_fb->p_data_end = p_fb->p_data_cur = p_fb->p_buf;
    p_fb->p_buf_end = p_fb->p_buf + n_byte_buf_init;

    /* p_fb->p_obj_undefined since no object yet */

    /* Open file. It is opened in binary mode because the scanning will
     * handle all EOL chars, so any translations by the OS are almost
     * pure overhead. Also, not converting ensures that if fread returns
     * fewer than the requested number of chars, that read was to the
     * end of the file (if not an error). Otherwise an additional read
     * getting a size of 0 would be required. */
    if ((p_fb->fp = fopen(filename, "rb")) == (FILE *) NULL) {
        xrc = -1;
        goto EXITPOINT;
    }

EXITPOINT:
    /* Free resources on error */
    if (xrc != 0) {
        if (p_fb != (FILEBUF *) NULL) {
            const int errno_save = errno; /* save errno in case fbclose()
                                           * changes it */
            (void) fbclose(p_fb);
            errno = errno_save;
            p_fb = (FILEBUF *) NULL;
        }
    } /* end of case of error */

    return p_fb;
} /* end of function fbopen */



/* This function frees resources used by a FILEBUF.
 *
 * Parameter
 * p_fb: The address of the FILEBUF to free. This argument may be NULL.
 *
 * Return values
 * 0: OK
 * EOF: Error closing file. Details can be found using errno.
 */
int fbclose(FILEBUF *p_fb)
{
    if (p_fb == (FILEBUF *) NULL) {
        return 0;
    }

    int xrc = 0;

    {
        void *p;
        if ((p = p_fb->p_buf) != NULL) {
            free(p);
        }
    }

    {
        FILE *fp;
        if ((fp = p_fb->fp) != (FILE *) NULL) {
            xrc = fclose(fp);
        }
    }

    free(p_fb);
    return xrc;
} /* end of function fbclose */



/* This function gets the next object converting it to the most desired
 * type.
 *
 * Parameters
 * p_fb: FILEBUF pointer initialized using fbopen()
 * n_type_wanted: number of desired type conversions for data from highest
 *      priority to lowest.
 * p_type_wanted: Desired type conversions for data from highest priority
 *      to lowest.
 * p_type_found: Address to receive the type of the data obtained
 * p_fbobj: Address of an FBOBJ structure to receive the data
 *
 * Return codes
 * +1: EOF reached
 * 0: Normal return
 * -1: Error. Use errno for further details.
 *
 * Remarks
 * Type BUF_TYPE_STRING is always implicitly added to the list of wanted
 *      types as the final choice, which any data will satisfy
 *
 * A string may be double-quoted. In this case the quotes are not supplied
 * to the caller as part of the data. Double-quoting ensures that a string
 * will not be converted to any other type. Within double quotes, a double
 * qoute and a backslash are escaped by a backslash, and a final unescaped
 * double quote is impilcitly added if EOF is reached when scanning for a
 * closing quote.
 *
 * A "*" or a "#" not within a quoted expression begins a comment that
 * extends to the end of the line.
 *
 * When called p_fb has data from the last get or it is the first call.
 *
 * Return Codes
 * +1: EOF
 * 0: Normal
 * -1: Error
 */
int fbget(FILEBUF *p_fb,
        unsigned int n_type_wanted, FBTYPE *p_type_wanted,
        FBTYPE *p_type_found, FBOBJ *p_fbobj)
{
    /* Test for existing EOF */
    if (p_fb->is_eof && p_fb->p_data_cur == p_fb->p_data_end) { /* no data */
        return +1;
    }

    /* Init to no object */
    p_fb->p_obj_start = (char *) NULL;

    /* Skip the comment if the initiating character was processed during
     * the last call to fbget */
    if (p_fb->f_skip_to_eol) {
        const int rc = fb_skip_to_eol(p_fb);
        if (rc != 0) { /* EOF or error */
            return rc;
        }
    }

    {
        const int rc = fb_skip_whitespace(p_fb);
        if (rc != 0) { /* EOF or error */
            return rc;
        }
    }

    /* Current char exists and starts the item */
    if (*p_fb->p_data_cur == '"') { /* quoted string */
        return fbget_quoted_string(p_fb, p_type_found, p_fbobj);
    }

    /* Else unquoted string */
    return fbget_unquoted_string(p_fb, n_type_wanted, p_type_wanted,
            p_type_found, p_fbobj);
} /* end of function fbget */



/* Get a quoted string given at a quote. On entry p_fb->p_data_cur points
 * to the quote starting the quoted string. On return it points to the first
 * character after the current item or equals p_fb->p_data_end if the
 * current item extens to the end of the current data string. */
static int fbget_quoted_string(FILEBUF *p_fb,
        FBTYPE *p_type_found, FBOBJ *p_fbobj)
{
    /* Advance past the opening quote to the true start of the string */
    if (++p_fb->p_data_cur == p_fb->p_data_end) {
        /* The leading quote ended the current data */
        const int rc = fb_fill(p_fb); /* Try to refill buffer */
        if (rc != 0) { /* EOF or error */
            if (rc < 0) { /* error */
                return -1;
            }
            /* Else EOF. This item is an empty string that ended without the
             * closing quote, so add an implicit closing quote, i.e., end
             * the string to form "".
             *
             * Since the object was started at the beginning of the buffer
             * and the buffer has at leat 1 byte a NULL to create the
             * string "" can be written here */
            *(p_fb->p_obj_end = p_fb->p_obj_start = p_fb->p_buf) = '\0';
            return fb_return_string(p_fb, p_type_found, p_fbobj);
        }
        /* Else data is now available at p_fb->p_data_cur */
    } /* end of case that at end of data from file */

    /* Save the start of the string as the current position */
    p_fb->p_obj_start = p_fb->p_data_cur;

    /* Continue processing as an unescaped string, unless the contrary
     * is found to be true */
    return fbget_quoted_unescaped_string(p_fb, p_type_found, p_fbobj);
} /* end of function fbget_quoted_string */



/* Get a quoted string with no escape. The start has already been set on
 * entry. If an escape is found, processing continues as an escaped string */
int fbget_quoted_unescaped_string(FILEBUF *p_fb,
        FBTYPE *p_type_found, FBOBJ *p_fbobj)
{
    /* Step through characters until end or escape */
    char *p_data_cur = p_fb->p_data_cur;
    char *p_data_end = p_fb->p_data_end;
    for ( ; ; ) { /* continue until done */
        for ( ; p_data_cur != p_data_end; ++p_data_cur) { /* current data */
            const char ch_cur = *p_data_cur;
            if (ch_cur == '"') { /* Closing quote, so done */
                *(p_fb->p_obj_end = p_data_cur) = '\0';
                p_fb->p_data_cur = p_data_cur + 1;
                return fb_return_string(p_fb, p_type_found, p_fbobj);
            }
            if (ch_cur == '\\') { /* Escape */
                /* After an escape, data must be moved to fill in the gap
                 * left by the escape character */
                p_fb->p_data_cur = p_data_cur; /* Reprocess the escape */
                return fbget_quoted_escaped_string(p_fb,
                        p_type_found, p_fbobj);
            }
            /* Else the character is part of the quoted string */
        } /* end of loop over current text */

        p_fb->p_data_cur = p_data_cur; /* update current position */
        const int rc = fb_fill(p_fb); /* Try to refill buffer */
        if (rc != 0) { /* EOF or error */
            if (rc < 0) { /* error */
                return -1;
            }
            /* Else EOF. Ended without closing quote, so add an implicit
             * closing quote, i.e., end the string. Since fb_fill()
             * did not return -1, there is at least 1 byte at the end of
             * the buffer where the read would have gone. */
            *(p_fb->p_obj_end = p_fb->p_data_cur) = '\0';
            return fb_return_string(p_fb, p_type_found, p_fbobj);
        }
        p_data_cur = p_fb->p_data_cur; /* Update after fill */
        p_data_end = p_fb->p_data_end;
    } /* end of loop processing until done or escape */
} /* end of function fbget_quoted_unescaped_string */



/* Get a quoted string with an escape. The start has already been set on
 * entry */
static int fbget_quoted_escaped_string(FILEBUF *p_fb,
        FBTYPE *p_type_found, FBOBJ *p_fbobj)
{
    /* Step through characters until end */
    char *p_data_src = p_fb->p_data_cur; /* at current char */
    char *p_data_dst = p_data_src; /* at current char */
    char *p_data_end = p_fb->p_data_end;
    bool f_escape_in_progress = false;
    for ( ; ; ) { /* continue until done */
        for ( ; p_data_src != p_data_end; ++p_data_src) { /* current data */
            const char ch_cur = *p_data_src;
            if (f_escape_in_progress) { /* Always copy the char */
                f_escape_in_progress = false;
                *p_data_dst++ = ch_cur;
            }
            else { /* Not an escaped character */
                if (ch_cur == '"') { /* Closing quote, so done */
                    p_fb->p_data_cur = p_data_src + 1;
                    *(p_fb->p_obj_end = p_data_dst) = '\0';
                    return fb_return_string(p_fb, p_type_found, p_fbobj);
                }
                if (ch_cur == '\\') { /* Escape */
                    f_escape_in_progress = true;
                    /* Do not copy the escape or advancd p_data_dst */
                }
                else { /* ordinary character */
                    *p_data_dst++ = ch_cur;
                }
            } /* end of case of not an escaped character */
            /* Else the character is part of the quoted string */
        } /* end of loop over current text */

        /* Indicate that there is no more unprocessed data */
        p_fb->p_data_end = p_fb->p_data_cur = p_data_dst;

        /* If no pending escape, can switch back to unescaped version and
         * avoid the moves */
        if (!f_escape_in_progress) {
            return fbget_quoted_unescaped_string(p_fb, p_type_found,
                    p_fbobj);
        }

        /* Else escape must be processed, so continue with escaped version */
        const int rc = fb_fill(p_fb); /* Try to refill buffer */
        if (rc != 0) { /* EOF or error */
            if (rc < 0) { /* error */
                return -1;
            }
            /* Else EOF. Ended without closing quote, so add an implicit
             * closing quote, i.e., end the string. Since fb_fill()
             * did not return -1, there is at least 1 byte at the end of
             * the buffer where the read would have gone. */
            *(p_fb->p_obj_end = p_fb->p_data_cur) = '\0';
            return fb_return_string(p_fb, p_type_found, p_fbobj);
        }
        p_data_dst = p_data_src = p_fb->p_data_cur; /* Update after fill */
        p_data_end = p_fb->p_data_end;
    } /* end of loop processing until done or escape */
} /* end of function fbget_quoted_escaped_string */



/* Get an  unquoted string starting at the current position */
static int fbget_unquoted_string(FILEBUF *p_fb,
        unsigned int n_type_wanted, FBTYPE *p_type_wanted,
        FBTYPE *p_type_found, FBOBJ *p_fbobj)
{
    /* Save the start of the string as the current position */
    p_fb->p_obj_start = p_fb->p_data_cur;

    static const signed char p_map[1 << CHAR_BIT] = {
        [(unsigned char ) ' '] = (signed char) +1,
        [(unsigned char ) '\t'] = (signed char) +1,
        [(unsigned char ) '\n'] = (signed char) +1,
        [(unsigned char ) '\r'] = (signed char) +1,
        [(unsigned char ) '\v'] = (signed char) +1,
        [(unsigned char ) '\f'] = (signed char) +1,
        [(unsigned char ) '*'] = (signed char) -1,
        [(unsigned char ) '#'] = (signed char) -1
    };
    /* Step through characters until whitespace or comment */
    char *p_data_cur = p_fb->p_data_cur;
    char *p_data_end = p_fb->p_data_end;
    for ( ; ; ) { /* continue until done */
        for ( ; p_data_cur != p_data_end; ++p_data_cur) { /* current data */
            const char ch_cur = *p_data_cur;
            const signed char map_cur = p_map[(unsigned char) ch_cur];
            if (map_cur != 0) { /* ws or comment start, so done */
                *(p_fb->p_obj_end = p_data_cur) = '\0';
                p_fb->p_data_cur = p_data_cur + 1; /* 1st char past string */
                p_fb->f_skip_to_eol = map_cur < 0;
                return fb_return_obj(p_fb, n_type_wanted, p_type_wanted,
                        p_type_found, p_fbobj);
            }
            /* Else more of the string */
        } /* end of loop over current text */

        p_fb->p_data_cur = p_data_cur; /* update current position */
        const int rc = fb_fill(p_fb); /* Try to refill buffer */
        if (rc != 0) { /* EOF or error */
            if (rc < 0) { /* error */
                return -1;
            }
            /* Else EOF. Ended without closing quote, so add an implicit
             * closing quote, i.e., end the string. Since fb_fill()
             * did not return -1, there is at least 1 byte at the end of
             * the buffer where the read would have gone. */
            *(p_fb->p_obj_end = p_fb->p_data_cur) = '\0';
            return fb_return_obj(p_fb, n_type_wanted, p_type_wanted,
                    p_type_found, p_fbobj);
        }
        p_data_cur = p_fb->p_data_cur; /* Update after fill */
        p_data_end = p_fb->p_data_end;
    } /* end of loop processing until done or escape */
} /* end of function fbget_unquoted_string */



/* This function fills the buffer. First it moves all data in the interval
 * [start current object, current position) to the beginning of the buffer.
 *
 * If there is no space left at the end of the buffer after the move (so
 * that the move did not occur and data extends to the end), the buffer is
 * doubled in size. In either case, the end of the buffer is filled with
 * data read from the file. */
static int fb_fill(FILEBUF *p_fb)
{
    /* Exit if EOF already */
    if (p_fb->is_eof) {
        return +1;
    }

    /* Move the data in use to the front of the buffer if not already and
     * enlarge the buffer if still no space. Returned value is bytes
     * available at the end of the buffer at p_data_end */
    const size_t n_byte_read = fb_make_space_at_end(p_fb);
    if (n_byte_read == 0) {
        return -1;
    }

    const size_t n_got = fread(p_fb->p_data_end, 1, n_byte_read, p_fb->fp);
    if (n_got < n_byte_read) { /* EOF or error */
        if (ferror(p_fb->fp)) {
            return -1;
        }
        /* Else mark as EOF for subsequent calls */
        p_fb->is_eof = true;

        if (n_got == 0) { /* Nothing to return for this call */
            return +1;
        }
        /* Else partial buffer to return */
    }

    /* Extend the end of the data by the bytes obtained */
    p_fb->p_data_end += n_got;
    return 0;
} /* end of function fb_fill */



/* Make space at the end of the buffer by moving data of current object
 * to the front of the buffer and enlarging if there is still no room
 *
 * Return value: Number of spaces that are free at the end of p_buf on
 *      return. If 0, more space could not be obtained.
 */
static size_t fb_make_space_at_end(FILEBUF *p_fb)
{
     const char * const p_obj_start = p_fb->p_obj_start;
    const char *  const p_src = p_obj_start == (char *) NULL ?
            p_fb->p_data_cur : p_obj_start;
    char * const p_dst = p_fb->p_buf;

    /* Shift data in use to the front of the buffer if not already */
    if (p_dst != p_src) { /* object is not at start of buffer */
        const size_t n = (size_t)(p_fb->p_data_end - p_src);
        if (n > 0) { /* Will be 0 if skipping whitespace and comments */
            (void) memmove(p_dst, p_src, n);
        }

        /* Adjust pointers after the move */
        ptrdiff_t delta = p_src - p_dst;
        p_fb->p_data_cur -= delta;
        p_fb->p_data_end -= delta;

        if (p_obj_start != (char *) NULL) {
            p_fb->p_obj_start -= delta;
        }
        /* never called when p_obj_end is valid */
    }
    else { /* already at front */
        if (p_fb->p_buf_end - p_fb->p_data_end == 0) {
            const size_t n_alloc_orig = p_fb->n_byte_buf_alloc;

            /* For debugging, this added size can be made very small to
             * force many reallocs and to have strings end at "hard"
             * locations such as right before where a terminating null
             * should be added to a string */
            //const size_t n_added = 1;
            const size_t n_added = n_alloc_orig;

            const size_t n_byte_buf_alloc_new = n_alloc_orig + n_added;
            void * const p = realloc(p_fb->p_buf, n_byte_buf_alloc_new);
            if (p == NULL) {
                return 0;
            }
            /* Else allocation OK, so update buffer and internal pointers */
            ptrdiff_t delta = (char *) p - p_fb->p_buf;
            p_fb->p_buf = (char *) p;
            p_fb->p_buf_end = (char *) p + n_byte_buf_alloc_new;
            p_fb->n_byte_buf_alloc = n_byte_buf_alloc_new;
            p_fb->p_data_cur += delta;
            p_fb->p_data_end += delta;

            if (p_obj_start != (char *) NULL) {
                p_fb->p_obj_start += delta;
            }
            /* never called when p_obj_end is valid */
        }
    }

    return (size_t)(p_fb->p_buf_end - p_fb->p_data_end);
} /* end of function fb_make_space_at_end */



/* Skip whitespace, including comments starting at the current position */
static int fb_skip_whitespace(FILEBUF *p_fb)
{
    static const signed char p_map[1 << CHAR_BIT] = {
        [(unsigned char ) ' '] = (signed char) +1,
        [(unsigned char ) '\t'] = (signed char) +1,
        [(unsigned char ) '\n'] = (signed char) +1,
        [(unsigned char ) '\r'] = (signed char) +1,
        [(unsigned char ) '\v'] = (signed char) +1,
        [(unsigned char ) '\f'] = (signed char) +1,
        [(unsigned char ) '*'] = (signed char) -1,
        [(unsigned char ) '#'] = (signed char) -1
    };
    /* Step through characters until not whitespace (including comments) */
    char *p_data_cur = p_fb->p_data_cur;
    char *p_data_end = p_fb->p_data_end;
    for ( ; ; ) { /* continue until done */
        for ( ; p_data_cur != p_data_end; ++p_data_cur) { /* current data */
            const char ch_cur = *p_data_cur;
            const signed char map_cur = p_map[(unsigned char) ch_cur];
            if (map_cur == 0) { /* not in ws or at comment start, so done */
                p_fb->p_data_cur = p_data_cur;
                return 0;
            }
            if (map_cur == -1) { /* a comment has started */
                p_fb->p_data_cur = p_data_cur + 1; /* after comment start */
                const int rc = fb_skip_to_eol(p_fb);
                if (rc != 0) { /* EOF or error */
                    return rc;
                }

                /* Update local variables. Note that p_fb->p_data_cur is at
                 * the character after the comment, which is a \n or \r.
                 * These characters are whitespace that will be skipped,
                 * so incrementing past it in the ++p_data_cur of the for()
                 * only skips a character that will be skipped anyhow.
                 * (A long comment to say that
                 * p_data_cur = p_fb->p_data_cur - 1 is not necessary.) */
                p_data_cur = p_fb->p_data_cur;
                p_data_end = p_fb->p_data_end;
            } /* end of comment processing */
            /* Else whitespace, which is skipped */
        } /* end of loop over current text */

        p_fb->p_data_cur = p_data_cur; /* update current position */
        const int rc = fb_fill(p_fb); /* Try to refill buffer */
        if (rc != 0) { /* EOF or error */
            return rc;
        }
        /* Else got more text to process */
        p_data_cur = p_fb->p_data_cur; /* Update after fill */
        p_data_end = p_fb->p_data_end;
    } /* end of loop over text pieces */
} /* end of function fb_skip_whitespace */



/* Skip text to EOL char, starting at the current position */
static int fb_skip_to_eol(FILEBUF *p_fb)
{
    /* Step through characters until not whitespace (including comments) */
    char *p_data_cur = p_fb->p_data_cur;
    char *p_data_end = p_fb->p_data_end;
    for ( ; ; ) { /* continue until done */
        for ( ; p_data_cur != p_data_end; ++p_data_cur) { /* current data */
            const char ch_cur = *p_data_cur;
            if (ch_cur == '\n' || ch_cur == '\r') {
                p_fb->p_data_cur = p_data_cur;
                return 0;
            }
            /* Else not EOL, which is skipped */
        } /* end of loop over current text */

        p_fb->p_data_cur = p_data_cur; /* update current position */
        const int rc = fb_fill(p_fb); /* Try to refill buffer */
        if (rc != 0) { /* EOF or error */
            return rc;
        }
        /* Else got more text to process */
        p_data_cur = p_fb->p_data_cur; /* Update after fill */
        p_data_end = p_fb->p_data_end;
    } /* end of loop over text pieces */
} /* end of function fb_skip_to_eol */



/* Return the data found in the most preferred format possible */
static int fb_return_obj(FILEBUF *p_fb,
        unsigned int n_type_wanted, FBTYPE *p_type_wanted,
        FBTYPE *p_type_found, FBOBJ *p_fbobj)
{
    const char * const p_obj_start = p_fb->p_obj_start; /* data to convert */
    const char * const p_obj_end = p_fb->p_obj_end;

    /* Must test for null string separately since strto* does not set
     * errno in this case. Aside from that, it can only be returned
     * as a string anyhow. */
    if (p_obj_start != p_obj_end) { /* have a string besides "" */
        unsigned int i;
        for (i = 0; i < n_type_wanted; ++i) {
            FBTYPE type_cur = p_type_wanted[i];
            errno = 0;
            if (type_cur == BUF_TYPE_ULONG) {
                char *p_end;
                unsigned long val = strtoul(p_obj_start, &p_end, 10);
                /* Test for processing of full string. Note that checking
                 * for the end of the string rather than a NULL handles the
                 * case of an embedded NULL which the latter test would
                 * not */
                if (errno == 0 && p_end == p_obj_end) {
                    *p_type_found = BUF_TYPE_ULONG;
                    p_fbobj->ulong_value = val;
                    return 0;
                }
            }
            else if (type_cur == BUF_TYPE_LONG) {
                char *p_end;
                long val = strtol(p_obj_start, &p_end, 10);
                if (errno == 0 && p_end == p_obj_end) {
                    *p_type_found = BUF_TYPE_LONG;
                    p_fbobj->long_value = val;
                    return 0;
                }
            }
            else if (type_cur == BUF_TYPE_DOUBLE) {
                char *p_end;
                double val = strtod(p_obj_start, &p_end);
                if (errno == 0 && p_end == p_obj_end) {
                    *p_type_found = BUF_TYPE_DOUBLE;
                    p_fbobj->dbl_value = val;
                    return 0;
                }
            }
            else if (type_cur == BUF_TYPE_STRING) {
                break; /* exit loop and use default return of string */
            }
            else { /* unknown type */
                print_error("Unknown output data type %d is ignored.",
                        (int) type_cur);
            }
        } /* end of loop trying types */
    } /* end of case that string is not "" */

    /* If no rquested type was converted OK or string requested, return as
     * a string */
    return fb_return_string(p_fb, p_type_found, p_fbobj);
} /* end of function fb_return_obj */



/* Return string */
static int fb_return_string(FILEBUF *p_fb,
        FBTYPE *p_type_found, FBOBJ *p_fbobj)
{
    const char *p_data_start =
            p_fbobj->str_value.sz = p_fb->p_obj_start;
    p_fbobj->str_value.n_char = (size_t)(p_fb->p_obj_end - p_data_start);
    *p_type_found = BUF_TYPE_STRING;
    return 0;
} /* end of function fb_return_string */



