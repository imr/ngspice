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

//#define Strbig(n,a)   char a[n+4]={0, (char)Hi(n), (char)Lo(n)}
#define Strbig(n,a)   char*(a)=(char*)tmalloc((n+4)*sizeof(char)); \
                      (a)[0]=0; (a)[1]=(char)Hi(n); (a)[2]=(char)Lo(n)
#define Strdbig(n,a,b)   char*(a); char*(b); \
                      (a)=(char*)tmalloc((n+4)*sizeof(char)); \
                      (a)[0]=0; (a)[1]=(char)Hi(n); (a)[2]=(char)Lo(n);\
                      (b)=(char*)tmalloc((n+4)*sizeof(char)); \
                      (b)[0]=0; (b)[1]=(char)Hi(n); (b)[2]=(char)Lo(n)

#define Strfbig(n,a,b,c,d)   char*(a); char*(b); char*(c); char*(d);\
                      (a)=(char*)tmalloc((n+4)*sizeof(char)); \
                      (a)[0]=0; (a)[1]=(char)Hi(n); (a)[2]=(char)Lo(n);\
                      (b)=(char*)tmalloc((n+4)*sizeof(char)); \
                      (b)[0]=0; (b)[1]=(char)Hi(n); (b)[2]=(char)Lo(n);\
                      (c)=(char*)tmalloc((n+4)*sizeof(char)); \
                      (c)[0]=0; (c)[1]=(char)Hi(n); (c)[2]=(char)Lo(n);\
                      (d)=(char*)tmalloc((n+4)*sizeof(char)); \
                      (d)[0]=0; (d)[1]=(char)Hi(n); (d)[2]=(char)Lo(n)

#define Strrem(a)     tfree(a)
#define Strdrem(a,b)  tfree(a); tfree(b)
#define Strfrem(a,b,c,d)  tfree(a); tfree(b); tfree(c); tfree(d)

#define Str(n,a)      char a[n+3]={0,0,(char)n}  /* n<255 ! */
#define Sini(s)       sini(s,sizeof(s)-4)


/* was 255, then 15000, string maxlen, 40000 to catch really big 
   macros in .model lines, now just a big number, a line length
   which never should be exceeded, may be removed later*/
typedef enum {Maxstr=4000000} _nMaxstr;  
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
unsigned char naddll( char * s, long long n);
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
