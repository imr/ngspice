/*************
 * Header file for parse.c
 * 1999 E. Rouat
 ************/

#ifndef PARSE_H_INCLUDED
#define PARSE_H_INCLUDED

struct pnode * ft_getpnames(wordlist *wl, bool check);
void free_pnode(struct pnode *t);


#endif
