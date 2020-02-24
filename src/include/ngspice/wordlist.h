#ifndef ngspice_WORDLIST_H
#define ngspice_WORDLIST_H

#include <stdio.h>

/* Doubly linked lists of words. */
struct wordlist {
    char *wl_word;
    struct wordlist *wl_next;
    struct wordlist *wl_prev;
} ;

typedef struct wordlist wordlist;

wordlist * wl_append(wordlist *wlist, wordlist *nwl);
void wl_append_word(wordlist **first, wordlist **last, char *word);
wordlist * wl_build(const char * const *v);
wordlist *wl_chop(wordlist *wlist);
wordlist *wl_chop_rest(wordlist *wlist);
wordlist *wl_cons(char *word, wordlist *tail);
wordlist * wl_copy(const wordlist *wlist);
void wl_delete_slice(wordlist *from, wordlist *to);
wordlist *wl_find(const char *string, const wordlist *wlist);
char * wl_flatten(const wordlist *wl);
void wl_free(wordlist *wlist);
wordlist *wl_from_string(const char *sz);
int wl_length(const wordlist *wlist);
char ** wl_mkvec(const wordlist *wl);
wordlist * wl_nthelem(int i, wordlist *wl);
void wl_print(const wordlist *wlist, FILE *fp);
wordlist * wl_range(wordlist *wl, int low, int up);
wordlist * wl_reverse(wordlist *wl);
void wl_sort(wordlist *wl);
wordlist * wl_splice(wordlist *elt, wordlist *list);


#endif
