/*************
 * Header file for runcoms.c
 * 1999 E. Rouat
 * $Id$
 ************/

#ifndef RUNCOMS_H_INCLUDED
#define RUNCOMS_H_INCLUDED

void com_scirc(wordlist *wl);
void com_pz(wordlist *wl);
void com_op(wordlist *wl);
void com_dc(wordlist *wl);
void com_ac(wordlist *wl);
void com_tf(wordlist *wl);
void com_tran(wordlist *wl);
void com_sens(wordlist *wl);
void com_disto(wordlist *wl);
void com_noise(wordlist *wl);
void com_run(wordlist *wl);
int ft_dorun(char *file);
bool ft_getOutReq(FILE **fpp, struct plot **plotp, bool *binp, char *name, char *title);

extern FILE *rawfileFp;
extern bool rawfileBinary;
extern char *last_used_rawfile;
extern bool resumption;

#endif
