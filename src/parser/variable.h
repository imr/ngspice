/*************
 * Header file for variable.c
 * 1999 E. Rouat
 ************/

#ifndef VARIABLE_H_INCLUDED
#define VARIABLE_H_INCLUDED


wordlist * cp_varwl(struct variable *var);
void cp_vset(char *varname, char type, char *value);
struct variable * cp_setparse(wordlist *wl);
void cp_remvar(char *varname);



#endif
