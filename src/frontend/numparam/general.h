/*   general.h    */
/*
   include beforehand  the following:
#include <stdio.h>   // NULL FILE fopen feof fgets fclose fputs fputc gets
#include <stdlib.h>
   the function code is in 'mystring.c' .
*/
#include "ngspice/dstring.h"
#include "ngspice/bool.h"


void pscat(DSTRINGPTR s, const char *str, const char *stop);
void pscopy(DSTRINGPTR s, const char *str, const char *stop);
void scopyd(DSTRINGPTR dst, const DSTRINGPTR src);
void scopys(DSTRINGPTR a, const char *b);
void sadd(DSTRINGPTR s, const char *t);
void cadd(DSTRINGPTR s, char c);

bool alfa(char c);
bool alfanum(char c);
bool alfanumps(char c);

int yes_or_no(void);
