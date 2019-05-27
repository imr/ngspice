/*************
 * Header file for dimens.c
 * 1999 E. Rouat
 ************/

#ifndef ngspice_DIMENS_H
#define ngspice_DIMENS_H

void dimstring(const int *dim_data, int n_dim, char *retstring);
void indexstring(const int *dim_data, int n_dim, char *retstring);
int incindex(int *counts, int numcounts, const int *dims, int numdims);
int atodims(const char *p, int *data, int *outlength);

#ifdef COMPILE_UNUSED_FUNCTIONS
/* #ifdef COMPILE_UNUSED_FUNCTIONS added 2019-03-31 */
int emptydims(int *data, int length);
char *skipdims(char *p);
#endif

#endif
