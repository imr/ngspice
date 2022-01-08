/*************
 * Header file for runcoms.c
 * 1999 E. Rouat
 ************/

#ifndef ngspice_RUNCOMS_H
#define ngspice_RUNCOMS_H

void com_scirc(wordlist *wl);
void com_pz(wordlist *wl);
void com_op(wordlist *wl);
void com_dc(wordlist *wl);
void com_ac(wordlist *wl);
void com_tf(wordlist *wl);
void com_tran(wordlist *wl);
/* SP: Stady State Analysis */
void com_pss(wordlist *wl);
/* SP */
void com_sens(wordlist *wl);
void com_disto(wordlist *wl);
void com_noise(wordlist *wl);
void com_run(wordlist *wl);
#ifdef RFSPICE
void com_sp(wordlist* wl);
#endif

extern FILE *rawfileFp;
extern bool rawfileBinary;
extern char *last_used_rawfile;
extern bool resumption;

#endif
