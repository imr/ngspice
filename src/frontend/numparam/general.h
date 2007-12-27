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
#define Hi(x) (((x) >> 8) & 0xff)
#define Lo(x) ((x) & 0xff)

#define Strbig(n,a)   char a[n+4]={0, (char)Hi(n), (char)Lo(n)}
#define Str(n,a)      char a[n+3]={0,0,(char)n}  /* n<255 ! */
#define Sini(s)       sini(s,sizeof(s)-4)


typedef enum {Maxstr=25004} _nMaxstr;  /* was 255, string maxlen, may be up to 32000 or so */
typedef enum {Esc=27} _nEsc;
typedef enum {Tab=9} _nTab;
typedef enum {Bs=8} _nBs;
typedef enum {Lf=10} _nLf;
typedef enum {Cr=13} _nCr;

typedef char string[258];


void sini( char * s, int i);
void sfix(char * s, int i, int max);
int maxlen(char * s);
char * pscopy( char * s, char * a, int i,int j);
char * pscopy_up( char * s, char * a, int i,int j);
unsigned char scopy( char * a, char * b);
unsigned char scopy_up( char * a, char * b);
unsigned char ccopy( char * a, char c);
unsigned char sadd( char * s, char * t);
unsigned char nadd( char * s, long n);
unsigned char cadd( char * s, char c);
unsigned char sins( char * s, char * t);
unsigned char cins( char * s, char c);
int cpos( char c, char * s);
int spos( char * sub, char * s);
int ci_prefix( register char *p, register char *s );
int length(char * s);
unsigned char steq(char * s, char * t);
unsigned char stne(char * s, char * t);
int scompare(char * a, char * b);
int ord(char c);
int pred(int i);
int succ(int i);
void stri(long n, char * s);
void strif(long n, int f, char * s);
void strf(double x, int a, int b, char * s); /* float -> string */
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
void rs( char * s);
char rc(void);

int freadstr(FILE * f, char * s, int max);
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
char * newstring(int n);
