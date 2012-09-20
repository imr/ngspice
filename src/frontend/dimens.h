/*************
 * Header file for dimens.c
 * 1999 E. Rouat
 ************/

#ifndef DIMENS_H_INCLUDED
#define DIMENS_H_INCLUDED

void dimstring(int *data, int length, char *retstring);
void indexstring(int *data, int length, char *retstring);
int incindex(int *counts, int numcounts, int *dims, int numdims);
int emptydims(int *data, int length);
int atodims(char *p, int *data, int *outlength);
char *skipdims(char *p);

#endif
