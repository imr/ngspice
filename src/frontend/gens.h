/*************
 * Header file for gens.c
 * 1999 E. Rouat
 ************/

#ifndef GENS_H_INCLUDED
#define GENS_H_INCLUDED

void wl_forall(wordlist *wl, int (*fn) (/* ??? */), char *data);
dgen * dgen_init(GENcircuit *ckt, wordlist *wl, int nomix, int flag, int model);
int dgen_for_n(dgen *dg, int n, int (*fn) (/* ??? */), char *data, int subindex);
void dgen_nth_next(dgen **dg, int n);



#endif
