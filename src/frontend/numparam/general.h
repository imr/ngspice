/*   general.h    */
/*
   include beforehand  the following:
#include <stdio.h>   // NULL FILE fopen feof fgets fclose fputs fputc gets 
#include <stdlib.h>  
   the function code is in 'mystring.c' .
*/
#include "dstring.h"
#include "bool.h"

#define Use(x)  x=0;x=x
#define Uses(s) s=s 
#define Usep(x) x=x

typedef enum {Esc=27} _nEsc;
typedef enum {Tab=9} _nTab;
typedef enum {Bs=8} _nBs;
typedef enum {Lf=10} _nLf;
typedef enum {Cr=13} _nCr;

typedef char string[258];


void sfix( SPICE_DSTRINGPTR dstr_p, int len) ;
char * pscopy( SPICE_DSTRINGPTR s, char * a, int i,int j);
char * pscopy_up( SPICE_DSTRINGPTR s, char * a, int i,int j);
bool scopyd( SPICE_DSTRINGPTR a, SPICE_DSTRINGPTR b);
bool scopys( SPICE_DSTRINGPTR a, char *b);
bool scopy_up( SPICE_DSTRINGPTR a, char *str) ;
bool scopy_lower( SPICE_DSTRINGPTR a, char *str) ;
bool ccopy( SPICE_DSTRINGPTR a, char c);
bool sadd( SPICE_DSTRINGPTR s, char * t);
bool nadd( SPICE_DSTRINGPTR s, long n);
bool cadd( SPICE_DSTRINGPTR s, char c);
bool naddll( SPICE_DSTRINGPTR s, long long n);
bool cins( SPICE_DSTRINGPTR s, char c);
bool sins( SPICE_DSTRINGPTR s, char * t);
int cpos( char c, char *s);
int spos_( char * sub, char * s);
bool ci_prefix( register char *p, register char *s );
int length(char * s);
bool steq(char * s, char * t);
bool stne(char * s, char * t);
int scompare(char * a, char * b);
int ord(char c);
int pred(int i);
int succ(int i);
void stri(long n, SPICE_DSTRINGPTR s);
void strif(long n, int f, SPICE_DSTRINGPTR dstr_p);
void strf(double x, int a, int b, SPICE_DSTRINGPTR dstr_p); /* float -> string */
long   ival(char * s, int *err);
double rval(char * s, int *err);

char upcase(char c);
char lowcase(char c);
int hi(long w);
int lo(long w);
bool odd(long x);
bool alfa(char c);
bool num(char c);
bool alfanum(char c);
char * stupcase( char * s);

/***** primitive input-output ***/
void wc(char c);
void wln(void);
void ws( char * s);
void wi(long i);
void rs( SPICE_DSTRINGPTR s);
char rc(void);

int freadstr(FILE * f, SPICE_DSTRINGPTR dstr_p);
char freadc(FILE * f);
long freadi(FILE * f);

long np_round(double d);	// sjb to avoid clash with round() in math.h
long np_trunc(double x);	// sjb to avoid clash with trunc() in math.h
double sqr(double x);
double absf(double x); /* abs */
long absi( long i);
double frac(double x);

void rawcopy(void * a, void * b, int la, int lb);
void * new(long sz);
void dispose(void * p);
