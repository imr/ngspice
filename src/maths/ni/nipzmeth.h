/*************
 * Header file for nipzmeth.c
 * 1999 E. Rouat
 ************/

#ifndef NIPZMETH_H_INCLUDED
#define NIPZMETH_H_INCLUDED

int NIpzSym(PZtrial **set, PZtrial *new);
int NIpzComplex(PZtrial **set, PZtrial *new);
int NIpzMuller(PZtrial **set, PZtrial *newtry);
int NIpzSym2(PZtrial **set, PZtrial *new);

#endif
