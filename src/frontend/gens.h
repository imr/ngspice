/*************
 * Header file for gens.c
 * 1999 E. Rouat
 ************/

#ifndef GENS_H_INCLUDED
#define GENS_H_INCLUDED

#include "ngspice/dgen.h"

void wl_forall(wordlist *wl, void (*fn)(wordlist*, dgen*), dgen *data);
int dgen_for_n(dgen *dg, int n, int (*fn) (dgen*, IFparm*, int), IFparm *data, int subindex);
void dgen_nth_next(dgen **dg, int n);


#endif
