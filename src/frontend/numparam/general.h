/*   general.h    */
/*
   include beforehand  the following:
#include <stdio.h>   // NULL FILE fopen feof fgets fclose fputs fputc gets 
#include <stdlib.h>  
   the function code is in 'mystring.c' .
*/

#define Use(x)  x=0;x=x
#define Uses(s) s=s 
#define Usep(x) x=x

/* -----------------------------------------------------------------
 * This structure is modified from Tcl.   We do this to avoid a
 * conflict and later add a conditional compile to just use the Tcl
 * code if desired.
----------------------------------------------------------------- */
#define SPICE_DSTRING_STATIC_SIZE 200
typedef struct spice_dstring {
  char *string ;               /* Points to beginning of string:  either
			        * staticSpace below or a malloced array. */
  int length ;                 /* Number of non-NULL characters in the
				* string. */
  int spaceAvl ;               /* Total number of bytes available for the
			        * string and its terminating NULL char. */
  char staticSpace[SPICE_DSTRING_STATIC_SIZE] ;
			       /* Space to use in common case where string
				* is small. */
} SPICE_DSTRING, *SPICE_DSTRINGPTR ;

/* -----------------------------------------------------------------
 * spice_dstring_xxxx routines.  Used to manipulate dynamic strings.
----------------------------------------------------------------- */
extern void spice_dstring_init(SPICE_DSTRINGPTR dsPtr) ;
extern char *spice_dstring_append(SPICE_DSTRINGPTR dsPtr,char *string,int length) ;
extern char *spice_dstring_print(SPICE_DSTRINGPTR dsPtr,char *format, ... ) ;
extern char *spice_dstring_setlength(SPICE_DSTRINGPTR dsPtr,int length) ;
extern char *_spice_dstring_setlength(SPICE_DSTRINGPTR dsPtr,int length) ;
extern void spice_dstring_free(SPICE_DSTRINGPTR dsPtr) ;
#define spice_dstring_reinit(x_xz) spice_dstring_setlength(x_xz,0) ;
#define spice_dstring_value(x_xz) ((x_xz)->string)
#define spice_dstring_space(x_xz) ((x_xz)->spaceAvl)
#define spice_dstring_length(x_xz) ((x_xz)->length)

typedef enum {Esc=27} _nEsc;
typedef enum {Tab=9} _nTab;
typedef enum {Bs=8} _nBs;
typedef enum {Lf=10} _nLf;
typedef enum {Cr=13} _nCr;

typedef char string[258];


void sfix( SPICE_DSTRINGPTR dstr_p, int len) ;
char * pscopy( SPICE_DSTRINGPTR s, char * a, int i,int j);
char * pscopy_up( SPICE_DSTRINGPTR s, char * a, int i,int j);
unsigned char scopyd( SPICE_DSTRINGPTR a, SPICE_DSTRINGPTR b);
unsigned char scopys( SPICE_DSTRINGPTR a, char *b);
unsigned char scopy_up( SPICE_DSTRINGPTR a, char *str) ;
unsigned char scopy_lower( SPICE_DSTRINGPTR a, char *str) ;
unsigned char ccopy( SPICE_DSTRINGPTR a, char c);
unsigned char sadd( SPICE_DSTRINGPTR s, char * t);
unsigned char nadd( SPICE_DSTRINGPTR s, long n);
unsigned char cadd( SPICE_DSTRINGPTR s, char c);
unsigned char naddll( SPICE_DSTRINGPTR s, long long n);
unsigned char cins( SPICE_DSTRINGPTR s, char c);
unsigned char sins( SPICE_DSTRINGPTR s, char * t);
int cpos( char c, char *s);
int spos_( char * sub, char * s);
int ci_prefix( register char *p, register char *s );
int length(char * s);
unsigned char steq(char * s, char * t);
unsigned char stne(char * s, char * t);
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
unsigned char odd(long x);
unsigned char alfa(char c);
unsigned char num(char c);
unsigned char alfanum(char c);
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

unsigned char reset(FILE * f);
unsigned char rewrite(FILE * f);
void rawcopy(void * a, void * b, int la, int lb);
void * new(long sz);
void dispose(void * p);
