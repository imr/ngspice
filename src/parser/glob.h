/*************
 * Header file for glob.c
 * 1999 E. Rouat
 ************/

#ifndef GLOB_H_INCLUDED
#define GLOB_H_INCLUDED

wordlist * cp_doglob(wordlist *wlist);
char * cp_tildexpand(char *string);
bool cp_globmatch(char *p, char *s);


#endif
