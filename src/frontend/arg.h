/*************
 * Header file for arg.c
 * 1999 E. Rouat
 ************/

#ifndef ARG_H_INCLUDED
#define ARG_H_INCLUDED

char *prompt(FILE *fp);
int countargs(wordlist *wl);
wordlist *process(wordlist *wlist);
void arg_print(wordlist *wl, struct comm *command);
void arg_plot(wordlist *wl, struct comm *command);
void arg_load(wordlist *wl, struct comm *command);
void arg_let(wordlist *wl, struct comm *command);
void arg_set(wordlist *wl, struct comm *command);
void arg_display(void);
void outmenuprompt(char *string);


#endif
