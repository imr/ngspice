/*   general.h    */
/*
   include beforehand  the following:
#include <stdio.h>   // NULL FILE fopen feof fgets fclose fputs fputc gets
#include <stdlib.h>
   the function code is in 'mystring.c' .
*/
#include "ngspice/dstring.h"
#include "ngspice/bool.h"


typedef char string[258];


void sfix(SPICE_DSTRINGPTR dstr_p, int len);
char *pscopy(SPICE_DSTRINGPTR s, char *a, int i, int j);
char *pscopy_up(SPICE_DSTRINGPTR s, char *a, int i, int j);
bool scopyd(SPICE_DSTRINGPTR a, SPICE_DSTRINGPTR b);
bool scopys(SPICE_DSTRINGPTR a, char *b);
bool scopy_up(SPICE_DSTRINGPTR a, char *str);
bool scopy_lower(SPICE_DSTRINGPTR a, char *str);
bool ccopy(SPICE_DSTRINGPTR a, char c);
bool sadd(SPICE_DSTRINGPTR s, char *t);
bool nadd(SPICE_DSTRINGPTR s, long n);
bool cadd(SPICE_DSTRINGPTR s, char c);
bool naddll(SPICE_DSTRINGPTR s, long long n);
bool cins(SPICE_DSTRINGPTR s, char c);
bool sins(SPICE_DSTRINGPTR s, char *t);
int cpos(char c, char *s);
int spos_(char *sub, char *s);
bool ci_prefix(register char *p, register char *s);
int length(char *s);
bool steq(char *s, char *t);
bool stne(char *s, char *t);
void stri(long n, SPICE_DSTRINGPTR s);

char upcase(char c);
char lowcase(char c);
bool alfa(char c);
bool num(char c);
bool alfanum(char c);
char *stupcase(char *s);

/***** primitive input-output ***/
void rs(SPICE_DSTRINGPTR s);
char rc(void);

int freadstr(FILE *f, SPICE_DSTRINGPTR dstr_p);

long np_round(double d);        // sjb to avoid clash with round() in math.h
long np_trunc(double x);        // sjb to avoid clash with trunc() in math.h
double absf(double x); /* abs */
long absi(long i);

void *new(size_t sz);
void dispose(void *p);
