/*************
 * Header file for wlist.c
 * 1999 E. Rouat
 ************/

#ifndef WLIST_H_INCLUDED
#define WLIST_H_INCLUDED

int wl_length(wordlist *wlist);
void wl_free(wordlist *wlist);
wordlist * wl_copy(wordlist *wlist);
wordlist * wl_splice(wordlist *elt, wordlist *list);
void wl_print(wordlist *wlist, FILE *fp);
wordlist * wl_build(char **v);
char ** wl_mkvec(wordlist *wl);
wordlist * wl_append(wordlist *wlist, wordlist *nwl);
wordlist * wl_reverse(wordlist *wl);
char * wl_flatten(wordlist *wl);
wordlist * wl_nthelem(register int i, wordlist *wl);
void wl_sort(wordlist *wl);
wordlist * wl_range(wordlist *wl, int low, int up);




#endif
