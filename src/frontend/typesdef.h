/*************
 * Header file for typesdef.c
 * 1999 E. Rouat
 ************/

#ifndef TYPESDEF_H_INCLUDED
#define TYPESDEF_H_INCLUDED

void com_dftype(wordlist *wl);
char * ft_typabbrev(int typenum);
char * ft_typenames(int typenum);
int ft_typnum(char *name);
char * ft_plotabbrev(char *string);
void com_stype(wordlist *wl);


#endif
