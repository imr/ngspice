/*************
 * Header file for alias.c
 * 1999 E. Rouat
 ************/

#ifndef ALIAS_H_INCLUDED
#define ALIAS_H_INCLUDED

wordlist * cp_doalias(wordlist *wlist);
void cp_setalias(char *word, wordlist *wlist);
void cp_unalias(char *word);
void cp_paliases(char *word);
void com_alias(wordlist *wl);
void com_unalias(wordlist *wl);

#endif
