/*************
 * Header file for arg.c
 * 1999 E. Rouat
 ************/

#ifndef ngspice_ARG_H
#define ngspice_ARG_H

char *prompt(FILE *fp);
int countargs(wordlist *wl);
wordlist *process(wordlist *wlist);
void arg_print(wordlist *wl, struct comm *command);
void arg_plot(wordlist *wl, struct comm *command);
void arg_load(wordlist *wl, struct comm *command);
void arg_let(wordlist *wl, struct comm *command);
void arg_set(wordlist *wl, struct comm *command);
void arg_display(wordlist *wl, struct comm *command);
void outmenuprompt(char *string);


#endif
