/*************
 * Header file for cshpar.c
 * 1999 E. Rouat
 ************/

#ifndef CSHPAR_H_INCLUDED
#define CSHPAR_H_INCLUDED

wordlist * cp_parse(char *string);
void com_echo(wordlist *wlist);
wordlist * cp_redirect(wordlist *wl);
void cp_ioreset(void);
void com_shell(wordlist *wl);
void fixdescriptors(void);
void com_rehash(wordlist *wl);
void com_chdir(wordlist *wl);
void com_strcmp(wordlist *wl);


#endif
