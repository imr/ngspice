/*   general.h    */
/*
   include beforehand  the following:
#include <stdio.h>   // NULL FILE fopen feof fgets fclose fputs fputc gets
#include <stdlib.h>
   the function code is in 'mystring.c' .
*/
#include "ngspice/dstring.h"
#include "ngspice/bool.h"


char *pscopy(SPICE_DSTRINGPTR s, const char *str, const char *stop);
void scopyd(SPICE_DSTRINGPTR a, SPICE_DSTRINGPTR b);
void scopys(SPICE_DSTRINGPTR a, const char *b);
void sadd(SPICE_DSTRINGPTR s, const char *t);
void cadd(SPICE_DSTRINGPTR s, char c);

bool alfa(char c);
bool alfanum(char c);
bool alfanumps(char c);

int yes_or_no(void);
