/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Wayne A. Christopher, U. C. Berkeley CAD Group
**********/

/* Wordlist manipulation stuff.  */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ngspice/bool.h"
#include "ngspice/memory.h"
#include "ngspice/ngspice.h"
#include "ngspice/wordlist.h"


/* Determine the length of a word list. */
int
wl_length(const wordlist *wl)
{
    int i = 0;

    for (; wl; wl = wl->wl_next)
        i++;

    return (i);
}


/* Free the storage used by a word list. */
void
wl_free(wordlist *wl)
{
    while (wl) {
        wordlist *next = wl->wl_next;
        tfree(wl->wl_word);
        tfree(wl);
        wl = next;
    }
}


/* Copy a wordlist and the words. */
wordlist *
wl_copy(const wordlist *wl)
{
    wordlist *first = NULL, *last = NULL;

    for (; wl; wl = wl->wl_next)
        wl_append_word(&first, &last, copy(wl->wl_word));

    return (first);
}


/* Substitute a wordlist for one element of a wordlist, and return a
 * pointer to the last element of the inserted list.  */
wordlist *
wl_splice(wordlist *elt, wordlist *list)
{

    if (list)
        list->wl_prev = elt->wl_prev;
    if (elt->wl_prev)
        elt->wl_prev->wl_next = list;
    if (list) {
        while (list->wl_next)
            list = list->wl_next;
        list->wl_next = elt->wl_next;
    }
    if (elt->wl_next)
        elt->wl_next->wl_prev = list;
    tfree(elt->wl_word);
    tfree(elt);
    return (list);
}



static void
printword(const char *string, FILE *fp)
{
    if (string) {
        while (*string) {
            putc((*string++), fp);
        }
    }
}


/* Print a word list. (No \n at the end...) */
void
wl_print(const wordlist *wl, FILE *fp)
{
    for (; wl; wl = wl->wl_next) {
        printword(wl->wl_word, fp);
        if (wl->wl_next)
            putc(' ', fp);
    }
}


/* Turn an array of char *'s into a wordlist. */
wordlist *
wl_build(const char * const *v)
{
    wordlist *first = NULL;
    wordlist *last = NULL;

    while (*v)
        wl_append_word(&first, &last, copy(*v++));

    return first;
}



/* Convert a single string into a wordlist. */
wordlist *
wl_from_string(const char *sz)
{
    const char * list_of_1_word[2];
    list_of_1_word[0] = sz;
    list_of_1_word[1] = (char *) NULL;
    return wl_build(list_of_1_word);
} /* end of function wl_from_string */



char **
wl_mkvec(const wordlist *wl)
{
    int  len = wl_length(wl);
    char **vec = TMALLOC(char *, (size_t) len + 1);

    int i;

    for (i = 0; i < len; i++) {
        vec[i] = copy(wl->wl_word);
        wl = wl->wl_next;
    }
    vec[i] = NULL;

    return (vec);
}



/* Nconc two wordlists together. */
wordlist *
wl_append(wordlist *wlist, wordlist *nwl)
{
    wordlist *wl;
    if (wlist == NULL)
        return (nwl);
    if (nwl == NULL)
        return (wlist);
    for (wl = wlist; wl->wl_next; wl = wl->wl_next)
        ;
    wl->wl_next = nwl;
    nwl->wl_prev = wl;
    return (wlist);
}


/* Reverse a word list. */
wordlist *
wl_reverse(wordlist *wl)
{
    if (!wl)
        return (wl);

    for (;;) {
        SWAP(wordlist *, wl->wl_next, wl->wl_prev);
        if (!wl->wl_prev)
            return (wl);
        wl = wl->wl_prev;
    }
}


/* This function converts a wordlist into a string, adding a blank space
 * between each word and a null termination. The wordlist may be NULL, in
 * which case "" is returned.
 *
 * The returned string is allocated and must be freed by the caller. */
char *
wl_flatten(const wordlist *wlist)
{
    char *buf;
    const wordlist *wl;

    /* Handle case of an empty list */
    if (wlist == (wordlist *) NULL) {
        buf = TMALLOC(char, 1);
        *buf = '\0';
        return buf;
    }

    /* List has at least one word */

    /* Find size needed for buffer
     * +1 for interword blanks and null at end */
    size_t len = 0;
    for (wl = wlist; wl; wl = wl->wl_next)
        len += strlen(wl->wl_word) + 1;

    /* Allocate to min required size */
    buf = TMALLOC(char, len);

    /* Step through the list again, building the output string */
    char *p_dst = buf;
    for (wl = wlist; ; ) { /* for each word */
        /* Add all source chars until end of word */
        const char *p_src = wl->wl_word;
        for ( ; ; p_src++) { /* for each char */
            const char ch_src = *p_src;
            if (ch_src == '\0') { /* exit when null found */
                break;
            }
            *p_dst++ = ch_src;
        } /* end of loop over chars in source string */

        /* Move to next word, exiting if none left */
        if ((wl = wl->wl_next) == (wordlist *) NULL) {
            *p_dst = '\0'; /* null-terminate string */
            return buf; /* normal function exit */
        }
        *p_dst++ = ' '; /* add space between words */
    } /* end of loop over words in word list */
} /* end of function wl_flatten */



/* Return the nth element of a wordlist, or the last one if n is too
 * big.  Numbering starts at 0...  */
wordlist *
wl_nthelem(int i, wordlist *wl)
{
    while ((i-- > 0) && wl->wl_next)
        wl = wl->wl_next;

    return (wl);
}



/* Compare function for the array of word pointers */
static int
wlcomp(const char * const *s, const char * const *t)
{
    return strcmp(*s, *t);
}



/* Sort a word list in order of strcmp ascending */
void
wl_sort(wordlist *wl)
{
    size_t i = 0;
    wordlist *ww = wl;
    char **stuff;

    /* Find number of words in the list */
    for (i = 0; ww; i++) {
        ww = ww->wl_next;
    }

    /* If empty list or only one word, no sort is required */
    if (i <= 1) {
        return;
    }

    stuff = TMALLOC(char *, i); /* allocate buffer for words */

    /* Add pointers to the words to the buffer */
    for (i = 0, ww = wl; ww; i++, ww = ww->wl_next) {
        stuff[i] = ww->wl_word;
    }

    /* Sort the words */
    qsort(stuff, i, sizeof (char *),
            (int (*)(const void *, const void *)) &wlcomp);

    /* Put the words back into the word list in sorted order */
    for (i = 0, ww = wl; ww; i++, ww = ww->wl_next) {
        ww->wl_word = stuff[i];
    }

    tfree(stuff); /* free buffer of word pointers */
} /* end of function wl_sort */



/* Return a range of wordlist elements... */
wordlist *
wl_range(wordlist *wl, int low, int up)
{
    wordlist *tt;
    bool rev = FALSE;

    if (low > up) {
        SWAP(int, up, low);
        rev = TRUE;
    }
    up -= low;
    while (wl && (low > 0)) {
        tt = wl->wl_next;
        tfree(wl->wl_word);
        tfree(wl);
        wl = tt;
        if (wl)
            wl->wl_prev = NULL;
        low--;
    }
    tt = wl;
    while (tt && (up > 0)) {
        tt = tt->wl_next;
        up--;
    }
    if (tt && tt->wl_next) {
        wl_free(tt->wl_next);
        tt->wl_next = NULL;
    }
    if (rev)
        wl = wl_reverse(wl);
    return (wl);
}


/*
 *  prepend a new `word'
 *     to the front of the given `wlist' wordlist
 *  and return this new list
 */

wordlist *
wl_cons(char *word, wordlist *wlist)
{
    wordlist *w = TMALLOC(wordlist, 1);
    w->wl_next = wlist;
    w->wl_prev = NULL;
    w->wl_word = word;

    if (wlist)
        wlist->wl_prev = w;

    return (w);
}


/*
 * given a wordlist
 *   described by a `first' and `last' wordlist element
 * append a new `word'
 *   and update the given `first' and `last' pointers accordingly
 *
 * Remarks
 * Onwership of the buffer containing the word is given to the
 * word list. That is, the word is not copied.
 */

void
wl_append_word(wordlist **first, wordlist **last, char *word)
{
    wordlist *w = TMALLOC(wordlist, 1);
    w->wl_next = NULL;
    w->wl_prev = (*last);
    w->wl_word = word;

    if (*last)
        (*last)->wl_next = w;
    else
        (*first) = w;

    (*last) = w;
}


/*
 * given a pointer `wl' into a wordlist, cut off this list from its
 * preceding elements and return itself. Thus, the function creates two
 * valid word lists: the one before this word and the one starting with
 * this word and continuing to the end of the original word list.
 */
wordlist *
wl_chop(wordlist *wl)
{
    if (wl && wl->wl_prev) {
        wl->wl_prev->wl_next = NULL;
        wl->wl_prev = NULL;
    }
    return (wl);
}


/*
 * given a pointer `wl' into a wordlist
 *   cut off the rest of the list
 *   and return this rest
 */
wordlist *
wl_chop_rest(wordlist *wl)
{
    wordlist *rest = wl->wl_next;
    wl->wl_next = NULL;
    if(rest)
        rest->wl_prev = NULL;
    return (rest);
}


/*
 * search for a string in a wordlist
 */
wordlist *
wl_find(const char *string, const wordlist *wl)
{
    if (!string)
        return NULL;

    for (; wl; wl = wl->wl_next)
        if (eq(string, wl->wl_word))
            break;
    return ((wordlist *) wl);
} /* end of function wl_find */



/*
 * delete elements from a wordlist
 *   starting at `from'
 *   up to but exclusive of `to'.
 *   `to' may be NULL to delete from `from' to the end of the list
 *
 * Allocations for the deleted slice are freed.
 *
 * Note that the function does not check if `from' and `to' are in
 * the same word list initially or that the former precedes the latter.
 */
void
wl_delete_slice(wordlist *from, wordlist *to)
{

    if (from == to) { /* nothing to delete */
        return;
    }

    wordlist *prev = from->wl_prev;

    if (prev) {
        prev->wl_next = to;
    }

    if (to) {
        to->wl_prev->wl_next = NULL;
        to->wl_prev = prev;
    }

    wl_free(from);
} /* end of function wl_delete_slice */



