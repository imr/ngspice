#ifndef _WORDLIST_H
#define _WORDLIST_H


/* Doubly linked lists of words. */
struct wordlist {
    char *wl_word;
    struct wordlist *wl_next;
    struct wordlist *wl_prev;
} ;

typedef struct wordlist wordlist;

extern char **wl_mkvec();
extern char *wl_flatten();
extern int wl_length();
extern void wl_free();
extern void wl_print();
extern void wl_sort();
extern wordlist *wl_append();
extern wordlist *wl_build();
extern wordlist *wl_copy();
extern wordlist *wl_range();
extern wordlist *wl_nthelem();
extern wordlist *wl_reverse();
extern wordlist *wl_splice();

#endif
