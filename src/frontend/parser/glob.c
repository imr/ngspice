/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Wayne A. Christopher, U. C. Berkeley CAD Group
**********/

/*
 * Expand global characters.
 *
 * e.g. text substitution like requested in the following script
 * set text = "mytext"
 * set newtext = new.{$text}
 * echo $newtext
 *
 *
 */
#include <stdint.h>

#include "ngspice/ngspice.h"
#include "ngspice/cpdefs.h"
#include "ngspice/wordlist.h"
#include "../misc/tilde.h"
#include "glob.h"

#ifdef HAVE_SYS_DIR_H
#include <sys/types.h>
#include <sys/dir.h>
#else

#ifdef HAVE_DIRENT_H
#include <sys/types.h>
#include <dirent.h>
#ifndef direct
#define direct dirent
#endif
#endif

#endif

#ifdef HAVE_PWD_H
#include <pwd.h>
#endif

#define OPT_WLL_COPY_ALL    1
/* This structure is a "long-form" of the wordlist structure. The inital
 * wordlist structure fields have the same meanings as in a standalone
 * wordlist structure, except that the allocations for p_escape and
 * p_after are separate from the one for wl_word. This structure is useful
 * when a wordlist must undergo many modifications to its words, as when
 * globbing is being expanded */
typedef struct wordlist_l {
    struct wordlist wl;
    size_t n_char_word; /* length of word excluding null */
    size_t n_elem_word_alloc; /* Allocated size of word array */
} wordlist_l;

static void wl_modify_word(wordlist *wl_node, unsigned int n_input,
        const size_t *p_n_char_word, char **pp_worde);
static wordlist *wll_to_wl(const wordlist_l *wll);
static wordlist_l *wll_append(wordlist_l *wl_dst, wordlist_l *wl_to_append);
static void wll_append_to_node(wordlist_l *dst, const wordlist_l *to_append);
static wordlist_l *wll_cons(
        size_t n_elem_word_alloc, size_t n_char_word, const char *p_word,
        unsigned int opt, wordlist_l *tail);
static void wll_free(wordlist_l *wll);
static wordlist *wll_node_to_wl_node(const wordlist_l *wll);

char cp_comma = ',';
char cp_ocurl = '{';
char cp_ccurl = '}';
char cp_til = '~';

static wordlist_l *brac1(size_t offset_ocurl1, const char *p_str_cur);
static wordlist_l *brac2(const char *string,
        size_t *p_n_char_processed);
static wordlist *bracexpand(const wordlist *w_exp);
static inline void merge_home_with_rest(wordlist *wl_node,
        size_t n_char_home, const char *sz_home, size_t n_char_skip);
static inline void strip_1st_char(wordlist *wl_node);
static void tilde_expand_word(wordlist *wl_node);


/* For each word, go through two steps: expand the {}'s, and then do ?*[]
 * globbing in them. Sort after the second phase but not the first...
 *
 * Globbing of arbitrary levels of brace nesting and tilde expansion to the
 * name of a "HOME" directory are supported. ?*[] are not */
wordlist *cp_doglob(wordlist *wlist)
{
    /* Expand {a,b,c} */
    {
        wordlist *wl = wlist;
        while (wl != (wordlist *) NULL) {
            wordlist *w = bracexpand(wl);
            if (!w) {
                wl_free(wlist);
                return (wordlist *) NULL;
            }

            /* Replace the node that was just expanded, wl, with the
             * expansion w (if different) and continue after that */
            if (wl != w) {
                wordlist *wl_next = wl->wl_next;
                (void) wl_splice(wl, w);

                /* Update head of list if the replacement
                 * changed it */
                if (wlist == wl) {
                    wlist = w;
                }

                /* Continue after the spliced nodes since
                 * they are already fully expanded */
                wl = wl_next;
            }
            else { /* same node, so just step to the next node */
                wl = wl->wl_next;
            }
        } /* end of loop over words in wordlist */
    } /* end of block expanding braces */

    /* Do tilde expansion on each word. */
    {
        wordlist *wl;
        for (wl = wlist; wl; wl = wl->wl_next) {
            if (*wl->wl_word == cp_til) {
                tilde_expand_word(wl);
            }
        } /* end of loop over words in wordlist */
    } /* end of block expanding braces */

    return wlist;
} /* end of function cp_doglob */



static wordlist *bracexpand(const wordlist *w_exp)
{
    const char * const wl_word = w_exp->wl_word;

    /* If no string, nothing to expand */
    if (wl_word == (char *) NULL) {
        return (wordlist *) NULL;
    }


    /* Find first opening brace. If none, the string expands to itself as
     * a wordlist */
    size_t offset_ocurl = ~(size_t) 0; /* flag for not found */

    /* Loop until find opening brace or end of string */
    {
        const char *p_cur = wl_word;
        char ch_cur;
        for ( ; (ch_cur = *p_cur) != '\0'; p_cur++) {
            if (ch_cur == cp_ocurl) {
                offset_ocurl = (size_t) (p_cur - wl_word);
                break;
            }
        }
    }


    /* Test for '{' and glob if there is one */
    if (offset_ocurl != ~(size_t) 0) {
        /* Found a brace, so glob */
        wordlist_l *wll_glob = brac1(offset_ocurl, wl_word);

        wordlist *wl_glob = wll_to_wl(wll_glob);
        wll_free(wll_glob);
        return wl_glob;
    }

    /* Unescaped '{' not found, so return the input node */
    return (wordlist *) w_exp;
} /* end of function bracexpand */



/* Given a string, returns a wordlist of all the {} expansions. This function
 * calls cp_brac2() with braced expressions and is called recursively by
 * cp_brac2().
 *
 * Parameters
 * offset_ocurl1: Offset from p_str where the first opening brace occurs
 *      or the offset to the terminating null of p_str (length of the
 *      string) if it contains no opening brace.
 */
static wordlist_l *brac1(size_t offset_ocurl1, const char *p_str)
{
    wordlist_l *words;
    const char *s;

    /* Create the inital entry in the list using all of the characters
     * before the first '{' */
    {
        const size_t n_byte_alloc = BSIZE_SP + 1;

        words = wll_cons(n_byte_alloc, offset_ocurl1, p_str,
                OPT_WLL_COPY_ALL, (wordlist_l *) NULL);
    }

    /* Step through string. In each iteration one {} group and the ungrouped
     * characters following that group, if any are processed */
    for (s = p_str + offset_ocurl1; *s != '\0'; ) {
        { /* Process braced expression */
            size_t n_char_processed;

            /* Process braced list using brac2() */
            wordlist_l *nwl = brac2(s,
                    &n_char_processed);
            if (nwl == (wordlist_l *) NULL) {
                /* brac2() already printed an error message */
                wll_free(words);
                return (wordlist_l *) NULL;
            }

            /* New wordlist to replace existing words. Note
             * the number of nodes is
             * #(existing list) X #(brac2() list). Each of
             * the brac2() words is appended to each of the
             * existing words to form the new wordlist */
            wordlist_l *newwl = (wordlist_l *) NULL;

            /* For each word in the existing word list (words) */
            wordlist_l *wl; /* loop iterator */
            for (wl = words; wl; wl = (wordlist_l *) wl->wl.wl_next) {

                /* For each word in the word list from brac2() */
                wordlist_l *w; /* loop iterator */
                for (w = nwl; w; w = (wordlist_l *) w->wl.wl_next) {
                    wordlist_l *nw = wll_cons(
                            BSIZE_SP + 1, 0, (char *) NULL,
                            OPT_WLL_COPY_ALL, (wordlist_l *) NULL);
                    wll_append_to_node(nw, wl);
                    wll_append_to_node(nw, w);
                    newwl = wll_append(newwl, nw);
                } /* end of loop over words from brac2() */
            } /* end of loop over words */
            wll_free(words);
            wll_free(nwl);
            words = newwl;
            s += n_char_processed; /* skip braced list */
        } /* end of processing of braced expression */

        {
            /* Apend all chars after {} expression until the next
             * '{' or the end of the word to each word in the wordlist */
            const char * const p_start = s;
            char ch_cur;
            for ( ; (ch_cur = *s) != cp_ocurl; s++) {
                if (ch_cur == '\0') {
                    break;
                }
            }

            const size_t n_char_append = (size_t) (s - p_start);

            if (n_char_append > 0) {
                wordlist_l *wl;
                for (wl = words; wl; wl = (wordlist_l *) wl->wl.wl_next) {
                    const size_t n_char_total = wl->n_char_word +
                            n_char_append;
                    const size_t n_elem_needed = n_char_total + 1;
                    if (wl->n_elem_word_alloc < n_elem_needed) {
                        const size_t n_elem_alloc = 2 * n_elem_needed;
                        wl->wl.wl_word = TREALLOC(char, wl->wl.wl_word,
                                n_elem_alloc);
                        wl->n_elem_word_alloc = n_elem_alloc;
                    }
                    char *p_dst = wl->wl.wl_word + wl->n_char_word;
                    (void) memcpy(p_dst, p_start, n_char_append);
                    p_dst += n_char_append;
                    *p_dst = '\0';
                    wl->n_char_word = n_char_total;
                }
            }
        } /* end of characters after braced expression */
    } /* end of loop over braced expressions + following chars in string */

    return words;
} /* end of function brac1 */



/* Given a string starting with a {, return a wordlist of the expansions
 * for the text until the matching }. A vaild input string must have both
 * an opening brace and a closing brace. If an error occurs, NULL is
 * returned. On a successful return, *p_n_char_processed will contain the
 * number of characters processed by brac2 up to and including the closing
 * outermost brace
 */
static wordlist_l *brac2(const char *string,
        size_t *p_n_char_processed)
{
    wordlist_l *wlist = (wordlist_l *) NULL;
    char buf_fixed[BSIZE_SP]; /* default work buffer */
    char *buf = buf_fixed; /* actual work buffer */
    bool eflag = FALSE; /* end-of-processing flag */

    /* Required buffer size. Note that 1st char of string is not copied,
     * so strlen(string) includes the length of the null at the end */
    const size_t n_elem_needed = strlen(string);

    /* Allocate and use a larger buffer if required */
    if (n_elem_needed > BSIZE_SP) { /* will not fit in stack buffer */
        buf = TMALLOC(char, n_elem_needed);
    }

    string++;   /* Get past the first open brace... */
    (void) strcpy(buf, string); /* make a copy of string */
    char *buf_cur = buf; /* current position in buffer */

    /* Each iteration of the outer loop processes one comma-separated
     * expression of the top-level brace-enclosed list */
    for ( ; ; ) {
        int nb = 0; /* number of braces **inside 1st brace** */
        size_t offset_ocurl1 = SIZE_MAX; /* Offset to 1st '{' */

        /* Start processing at start of next top-level term */
        char *s = buf_cur;

        /* Scan the string until the next comma at the top level is found,
         * the closing brace at the top level is found or the string ends
         * with a missing brace. If another term is found, it is processed
         * as a null-terminated string by calling brac1() */
        for ( ; ; ) {
            const char ch_cur = *s;
            if (ch_cur == cp_ccurl) { /* closing brace found */
                if (nb == 0) {
                    /* A closing brace found without any internal opening
                     * braces, so this one is the outermost brace */
                    eflag = TRUE; /* done -- set end flag */
                    break;
                }
                /* Else closing brace of internal level */
                nb--;
            }
            else if (ch_cur == cp_ocurl) { /* another brace level started */
                if (nb++ == 0) { /* Inc count. If 1st '{', save offset */
                    offset_ocurl1 = (size_t) (s - buf_cur);
                }
            }
            else if ((ch_cur == cp_comma) && (nb == 0)) {
                /* Comma found outside of any internal braced
                 * expression */
                break;
            }

            /* Check if reached end of string. If so, the closing
             * brace was not present. */
            if (ch_cur == '\0') {
                fprintf(cp_err, "Error: missing }.\n");
                if (buf != buf_fixed) { /* free allocation if made */
                    txfree(buf);
                }
                if (wlist != (wordlist_l *) NULL) {
                    wll_free(wlist);
                }

                return (wordlist_l *) NULL;
            }
            s++; /* process next char */
        } /* end of loop finding end of braces */

        /* The above loop exits without returning from the function if either
         * a comma at the top level or the closing top level brace is reached.
         * Variable s points to this location. By setting it to null, a string
         * is created for one of the comma-separated expressions at the top
         * level */
        *s = '\0';

        /* Process the top-level expression found and append to the
         * wordlist being built */
        {
            wordlist_l *nwl = brac1(
                    offset_ocurl1 == SIZE_MAX ?
                            (size_t) (s - buf_cur) : offset_ocurl1,
                    buf_cur);
            wlist = wll_append(wlist, nwl);
        }

        /* Check for competion of processing */
        if (eflag) { /* done -- normal function exit */
            if (buf != buf_fixed) { /* free allocation if made */
                txfree(buf);
            }

            /* When the loop is exited, s is at a brace or comma, which
             * is also considered to be processed. Hence +2 not +1. */
            *p_n_char_processed = (size_t) (s - buf + 2);
            return wlist;
        }

        buf_cur = s + 1; /* go to next term after comma */
    } /* end of loop over top-level comma-separated expressions */
} /* end of function brac2 */



/* Expand tildes. */
char *cp_tildexpand(const char *string)
{
    /* Attempt to do the tilde expansion */
    char * const result = tildexpand(string);

    if (!result) { /* expansion failed */
        if (cp_nonomatch) { /* If set, should return the original string */
            return copy(string);
        }
        /* Else should return NULL to indiciate failure */
        return (char *) NULL;
    }

    return result; /* successful expansion returned */
} /* end of function cp_tildexpand */


/* This function expands the leading ~ of wl_node. */
static void tilde_expand_word(wordlist *wl_node)
{
    char *word = wl_node->wl_word;
    char *p_char_cur = ++word;
    char ch = *p_char_cur;
    if (ch ==  '\0' || ch ==  DIR_TERM) {
        char *sz_home;
        const int n_char_home = get_local_home(0, &sz_home);
        if (n_char_home < 0) { /* expansion failed */
            /* Strip the ~ and return the rest */
            strip_1st_char(wl_node);
            return;
        }
        merge_home_with_rest(wl_node, (size_t) n_char_home, sz_home, 1);
        return;
    }

#ifdef HAVE_PWD_H
    /* ~bob -- Get name of user and find home for that user */
    {
        char * const usr_start = wl_node->wl_word + 1;
        char *usr_end = usr_start;
        char c;
        while ((c = *usr_end) != '\0'  && c != DIR_TERM) {
            ++usr_end;
        }
        const size_t n_char_usr = (size_t) (usr_end - usr_start);
        const size_t n_byte_usr = n_char_usr + 1;
        const char c_orig = c; /* save char to be overwritten by '\0' */
        *usr_end = '\0';

        char *sz_home;
        const int n_char_home = get_usr_home(usr_start, 0, &sz_home);
        *usr_end = c_orig; /* restore char overwritten by '\0' */
        if (n_char_home < 0) {
            strip_1st_char(wl_node);
            return; /* Strip the ~ and return the rest */
        }
        merge_home_with_rest(wl_node, (size_t) n_char_home, sz_home,
                n_byte_usr);
        return;
    }

#else
    /* ~bob is meaningless. Strip the ~ and return the rest */
    strip_1st_char(wl_node);
    return;
#endif
} /* end of function tilde_expand_word */



/* Strip the 1st char. Equivalent to merging an empty HOME string with the
 * chars after the 1st char. Assumes string is at least 1 char long
 * excluding trailing NULL */
static inline void strip_1st_char(wordlist *wl_node)
{
    merge_home_with_rest(wl_node, 0, (char *) NULL, 1);
    return;
} /* end of function strip_1st_char */



/* This function modifies the wordlist node to consist of <home> +
 * rest of string after n_char_skip. It is assumed that the string
 * is at least n_char_skip characters long excluding the trailing null */
static inline void merge_home_with_rest(wordlist *wl_node,
        size_t n_char_home, const char *sz_home, size_t n_char_skip)
{
    size_t p_n_char_word[2];
    p_n_char_word[0] = n_char_home;
    p_n_char_word[1] = strlen(wl_node->wl_word) - n_char_skip;

    char *pp_word[2];
    pp_word[0] = (char *) sz_home;
    pp_word[1] = wl_node->wl_word + n_char_skip;

    wl_modify_word(wl_node, 2u, p_n_char_word, pp_word);
    return;
} /* end of function merge_home_with_rest */




/*** Long-form wordlist functions ***/

/* This function converts long-form wordlist wll to a standard wordlist.
 * The input long-form list is not modified. */
static wordlist *wll_to_wl(const wordlist_l *wll)
{
    /* Handle degnerate case of NULL input */
    if (wll == (wordlist_l *) NULL) {
        return (wordlist *) NULL;
    }

    /* There is at least one node in the long-form wordlist. */
    /* Convert it to a standard wordlist node, which is the node to
     * return */
    wordlist * const wl_start = wll_node_to_wl_node(wll);
    wordlist * wl_dst_prev = wl_start;
    wl_start->wl_prev = (wordlist *) NULL;

    /* Continue adding nodes */
    for (wll = (wordlist_l *) wll->wl.wl_next ; wll != (wordlist_l *) NULL;
            wll = (wordlist_l *) wll->wl.wl_next) {
        /* Convert a single long-form node to a standard for node */
        wordlist *wl_dst_cur = wll_node_to_wl_node(wll);
        wl_dst_prev->wl_next = wl_dst_cur;
        wl_dst_cur->wl_prev = wl_dst_prev;
        wl_dst_prev = wl_dst_cur;
    } /* end of loop over nodes in input long-form wordlist */

    /* Terminate the list of words */
    wl_dst_prev->wl_next = (wordlist *) NULL;
    return wl_start;
} /* end of function wll_to_wl */



/* This function creates word data for a standard list from word data of
 * a long-form list. Both structures must be allocated on input, and the
 * long-form data is not change */
static wordlist *wll_node_to_wl_node(const wordlist_l *wll)
{
    /* Allocate node being returned */
    wordlist * const wl_dst = TMALLOC(wordlist, 1);

    /* Find required size of allocation and save lengths of arrays */
    const size_t n_char_word = wll->n_char_word;
    const size_t n_byte_alloc = n_char_word + 1;

    /* Allocate buffer */
    char * p_dst_cur = wl_dst->wl_word = TMALLOC(char, n_byte_alloc);

    /* Word data */
    wl_dst->wl_word = p_dst_cur;
    (void) memcpy(p_dst_cur, wll->wl.wl_word, n_char_word);
    p_dst_cur += n_char_word;
    *p_dst_cur++ = '\0';

    return wl_dst;
} /* end of function wll_node_to_wl_node */



/* Free a long-form wordlist */
void wll_free(wordlist_l *wll)
{
    while (wll != (wordlist_l *) NULL) {
        wordlist_l * const next = (wordlist_l *) wll->wl.wl_next;
        void *p;
        if ((p = (void *) wll->wl.wl_word) != NULL) {
            txfree(p);
        }
        txfree(wll);
        wll = next;
    } /* end of loop over wordlist nodes */
} /* end of function wll_free */



/* This function prepends a wordlist_l node to the existing list.
 *
 * Parameters
 * n_char_word: Length of word, excluding trailing NULL
 * p_word: Address of word, or NULL if none. A null terminiation is
 *      not required if the word will be duplicated.
 * opt: OPT_WLL_COPY_ALL -- create copy instead of resuing word allocation
 * tail: Address of wordlist having this word prepended. May be null.
 *
 * Return value
 * New wordlist node which is the start of the list
 */
wordlist_l *wll_cons(
        size_t n_elem_word_alloc, size_t n_char_word, const char *p_word,
        unsigned int opt, wordlist_l *tail)
{
    /* Create a new node and link with the existing wordlist */
    wordlist_l *w = TMALLOC(wordlist_l, 1);
    w->wl.wl_next = (wordlist *) tail;
    w->wl.wl_prev = (wordlist *) NULL;

    w->n_char_word = n_char_word;
    w->n_elem_word_alloc = n_elem_word_alloc;

    if (opt & OPT_WLL_COPY_ALL) {
        char *p_dst = w->wl.wl_word = TMALLOC(char, n_elem_word_alloc);
        (void) memcpy(p_dst, p_word, n_char_word);
        p_dst += n_char_word;
        *p_dst = '\0';
    }
    else {
        w->wl.wl_word = (char *) p_word;
    }

    /* Link to front of rest of nodes, if present */
    if (tail) {
        /* The new word goes in the front */
        tail->wl.wl_prev = (wordlist *) w;
    }

    return w;
} /* end of function wl_cons */



/* This function appends wl_to_append to the end of wl_dst and returns the
 * start of the combined wordlist */
static wordlist_l *wll_append(wordlist_l *wl_dst, wordlist_l *wl_to_append)
{
    /* Handle degenerate cases where both of the input wordlists 
     * are not non-NULL */
    if (wl_dst == (wordlist_l *) NULL) {
        return wl_to_append;
    }
    if (wl_to_append == (wordlist_l *) NULL) {
        return wl_dst;
    }

    /* Locate last node of wl_dst */
    {
        wordlist_l *wl;
        for (wl = wl_dst; wl->wl.wl_next;
                wl = (wordlist_l *) wl->wl.wl_next) {
            ;
        }

        /* Link nwl to end */
        wl->wl.wl_next = (wordlist *) wl_to_append;
        wl_to_append->wl.wl_prev = (wordlist *) wl;
    }

    return wl_dst; /* Return combined wordlist */
} /* end of function wll_append */



/* This function appends word data, of the wordlist_l node "to_append" to
 * the existing values at wordlist_l node "dst".
 */
void wll_append_to_node(wordlist_l *dst, const wordlist_l *to_append)
{
    /* Get sizes */
    const size_t n_old = dst->n_char_word;
    const size_t n_new = to_append->n_char_word;
    const size_t n_total = n_old + n_new;
    const size_t n_elem_needed = n_total + 1;

    /* Resize if needed */
    if (dst->n_elem_word_alloc < n_elem_needed) {
        const size_t n_elem_alloc = 2 * n_elem_needed;
        dst->wl.wl_word = TREALLOC(
                char, dst->wl.wl_word, n_elem_alloc);
        dst->n_elem_word_alloc = n_elem_alloc;
    }

    /* Do append */
    {
        char *p_dst = dst->wl.wl_word + n_old;
        char * const p_src = to_append->wl.wl_word;
        (void) memcpy(p_dst, p_src, n_new);
        p_dst += n_new;
        *p_dst = '\0';
    }
    dst->n_char_word = n_total;
} /* end of function wll_append_to_node */



/* This function modifies word data in a word list node by building a new word
 * from the arrays supplied. These arrays may contain pieces of the word
 * being modified, with overlap and duplication of intervals allowed. Null
 * terminiations are not required on the input data.
 *
 * Parameters
 * wl_node: wordlist node being modified. Cannot be NULL.
 * n_input: Number of inputs
 * p_n_char_word: Array of length n_input of lengths of input strings,
        excluding trailing nulls
 * pp_word: Array of pointers to input character data. An entry may be NULL
 *      iff the corrseponding value of p_n_char_word is 0.
 */
static void wl_modify_word(wordlist *wl_node, unsigned int n_input,
        const size_t *p_n_char_word, char **pp_word)
{
    /* Find the number of chars of word data */
    size_t n_char_word_new = 0;
    { /* have array */
        /* Accumulate count of chars */
        const size_t *p_n_char_word_cur = p_n_char_word;
        const size_t * const p_n_char_word_end = p_n_char_word + n_input;
        for ( ; p_n_char_word_cur != p_n_char_word_end;
                ++p_n_char_word_cur) {
            n_char_word_new += *p_n_char_word_cur;
        }
    }

    /* New allocation */
    char *p_word_new;

    /* Process the segments. */
    { /* no escapes */
        /* + 1 for null after word */
        const size_t n_byte_alloc = n_char_word_new + 1;

        /* New allocation */
        p_word_new = TMALLOC(char, n_byte_alloc);

        /* New word. Build from input pieces */
        {
            const size_t *p_n_char_word_cur = p_n_char_word;
            const size_t * const p_n_char_word_end = p_n_char_word + n_input;
            char **pp_word_cur = pp_word;
            char *p_dst = p_word_new;
            for ( ; p_n_char_word_cur < p_n_char_word_end;
                    ++p_n_char_word_cur, ++pp_word_cur) {
                const size_t n_char_word_cur = *p_n_char_word_cur;
                (void) memcpy(p_dst, *pp_word_cur, n_char_word_cur);
                p_dst += n_char_word_cur;
            }
            *p_dst = '\0';
        }
    }


    /* Free old and assign new */
    txfree(wl_node->wl_word);
    wl_node->wl_word = p_word_new;
} /* end of function wl_modify_word */






