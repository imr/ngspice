/*************
 * Header file for parse.c
 * 1999 E. Rouat
 ************/

#ifndef _PARSE_H
#define _PARSE_H

#include <pnode.h>
#include <wordlist.h>

struct pnode * ft_getpnames(wordlist *wl, bool check);
void free_pnode(struct pnode *t);


#endif
