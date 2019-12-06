/*************
 * Header file for printnum.c
 * 1999 E. Rouat
 ************/

#ifndef ngspice_PRINTNUM_H
#define ngspice_PRINTNUM_H

#include "ngspice/dstring.h"

void printnum(char *buf, double num);
int printnum_ds(DSTRING *p_dstring, double num);

#endif
