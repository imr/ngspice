/*************
 * Header file for var2.c
 * 1999 E. Rouat
 ************/

#ifndef VAR2_H_INCLUDED
#define VAR2_H_INCLUDED


struct xxx {
    struct variable *x_v;
    char x_char;
} ;

wordlist * cp_variablesubst(wordlist *wlist);
wordlist * vareval(char *string);
void cp_vprint(void);
void com_set(wordlist *wl);
void com_unset(wordlist *wl);
void com_shift(wordlist *wl);
bool cp_getvar(char *name, int type, char *retval);




#endif
