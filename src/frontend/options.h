/*************
 * Header file for options.c
 * 1999 E. Rouat
 ************/

#ifndef OPTIONS_H_INCLUDED
#define OPTIONS_H_INCLUDED

struct variable * cp_enqvar(char *word);
void cp_usrvars(struct variable **v1, struct variable **v2);
struct line * inp_getopts(struct line *deck);
int cp_usrset(struct variable *var, bool isset);


#endif
