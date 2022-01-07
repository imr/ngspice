/*************
 * Header file for arg.c
 * 1999 E. Rouat
 ************/

#ifndef ngspice_ARG_H
#define ngspice_ARG_H

char *prompt(FILE *fp);
wordlist *process(wordlist *wlist);
void arg_print(const wordlist *wl, const struct comm *command);
void arg_plot(const wordlist *wl, const struct comm *command);
void arg_load(const wordlist *wl, const struct comm *command);
void arg_let(const wordlist *wl, const struct comm *command);
void arg_set(const wordlist *wl, const struct comm *command);
void arg_display(const wordlist *wl, const struct comm *command);
void arg_enodes(const wordlist *wl, const struct comm *command);
void outmenuprompt(const char *string);


#endif
