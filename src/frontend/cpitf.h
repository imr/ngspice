/*************
 * Header file for cpitf.c
 * 1999 E. Rouat
 ************/

#ifndef CPITF_H_INCLUDED
#define CPITF_H_INCLUDED

void ft_cpinit(void);
bool cp_istrue(wordlist *wl);
void cp_periodic(void);
void cp_doquit(void);
bool cp_oddcomm(char *s, wordlist *wl);


#endif
