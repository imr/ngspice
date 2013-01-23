/*************
 * Header file for breakp2.c
 * 1999 E. Rouat
 ************/

#ifndef ngspice_BREAKP2_H
#define ngspice_BREAKP2_H


void com_save(wordlist *wl);
void com_save2(wordlist *wl, char *name);
void settrace(wordlist *wl, int what, char *name);

extern struct dbcomm *dbs;
extern int debugnumber;

#endif
