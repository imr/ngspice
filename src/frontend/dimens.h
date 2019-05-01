/*************
 * Header file for dimens.c
 * 1999 E. Rouat
 ************/

#ifndef ngspice_DIMENS_H
#define ngspice_DIMENS_H

void dimstring(int *data, int length, char *retstring);
void indexstring(const int *dim_data, int n_dim, char *retstring);
int incindex(int *counts, int numcounts, int *dims, int numdims);
int emptydims(int *data, int length);
int atodims(char *p, int *data, int *outlength);
char *skipdims(char *p);

#endif
