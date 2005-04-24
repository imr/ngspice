/*   general.h    */

/*** Part 1: the C language redefined for quiche eaters ****
 *
 *  Real Hackers: undo all these macros with the 'washprog.c' utility ! 
 */

      /*  Proc ...... Begin .... EndProc  */
#define Proc void
#define Begin {
#define EndProc ;}
     /*  Func short ...(...) Begin...EndFunc   */
#define Func
#define EndFunc ;}
     /* If ... Then...ElsIf..Then...Else...EndIf */
#define If if(
#define Then ){
#define Else ;}else{
#define ElsIf ;}else if(
#define EndIf ;}
      /*  While...Do...Done */
#define While while(
#define Do ){
#define Done ;}
      /* Repeat...Until...EndRep */
#define Repeat do{
#define Until ;}while(!(
#define EndRep ));
     /*  For i=1;i<=10; Inc(i)  Do...Done  */
#define For for(
     /* Switch...CaseOne...Is...Case..Is...Default...EndSw */
#define Switch switch(
#define CaseOne ){ case
#define Case ;break;}case
#define AndCase :; case
#define Is :{
#define Default ;break;}default:{
#define EndSw ;break;}}

#define Record(x) typedef struct _t ## x {
#define RecPtr(x) typedef struct _t ## x *
#define EndRec(x) } x;
#define Addr(x) &x

#define False 0
#define True 1
#define Not !
#define And &&
#define Or ||
#define Div /
#define Mod %

#define Shl <<
#define Shr >>
#define AND &
#define OR  |
#define XOR ^
#define NOT ~
#define AT  *

#define Inc(p) (p)++
#define Dec(p) (p)--

/* see screened versions below:
#define New(t) (t*)malloc(sizeof(t))
#define Dispose(p) free((void*)p)
*/

#ifdef NULL
#define Null NULL
#else
#define Null (void *)0L
#endif

#define chr(x) (char)(x)
#define Zero(x) (!(x))
#define NotZ(x) (x)

typedef void* Pointer;
#define Type(a,b) typedef b a;

#ifdef _STDIO_H  /* somebody pulled stdio */
Type(Pfile, FILE AT)
#else
#ifdef __STDIO_H  /* Turbo C */
  Type(Pfile, FILE AT)
#else
  Type(Pfile, FILE*)   /* sjb - was Pointer, now FILE* */
#endif
#endif

Type(Char, unsigned char)
Type(Byte, unsigned char)
#ifndef Bool
Type(Bool, unsigned char)
#endif
Type(Word, unsigned short)
Type(Pchar, char AT)

#define Intern static
#define Extern extern
#define Tarray(a,d,n)     typedef d a[n];
#define Tarray2(a,d,n,m)  typedef d a[n][m];
#define Darray(a,d,n)     d a[n];

#define Const(x,y)    const short x=y;
#define Cconst(x,y)   typedef enum {x=y} _n ## x;

#define Aconst(a,tp,sze)  tp a[sze] ={
#define EndAco        };

/* the following require the 'mystring' mini-library */

#define Mcopy(a,b)    rawcopy((Pchar)a, (Pchar)b, sizeof(a),sizeof(b))
#define Rcopy(a,b)    rawcopy((Pchar)(&a), (Pchar)(&b), sizeof(&a),sizeof(&b))
#define New(tp)       (tp *)new(sizeof(tp))
#define Dispose(p)    dispose((void *)p)
#define NewArr(t,n)   (t *)new(sizeof(t)*n)


/*** Part 2: common 'foolproof' string library  ******/
/*
   include beforehand  the following:
#include <stdio.h>   // NULL FILE fopen feof fgets fclose fputs fputc gets 
#include <stdlib.h>  
   the function code is in 'mystring.c' .
*/

#define Use(x)  x=0;x=x
#define Uses(s) s=s 
#define Usep(x) x=x
#define Hi(x) (((x) Shr 8) AND 0xff)
#define Lo(x) ((x) AND 0xff)

#define Strbig(n,a)   char a[n+4]={0, (char)Hi(n), (char)Lo(n)}
#define Str(n,a)      char a[n+3]={0,0,(char)n}  /* n<255 ! */
#define Sini(s)       sini(s,sizeof(s)-4)

Cconst(Maxstr,2004) /* was 255, string maxlen, may be up to 32000 or so */

typedef char string[258];

Cconst(Esc, 27)
Cconst(Tab, 9)
Cconst(Bs, 8)
Cconst(Lf, 10)
Cconst(Cr, 13)

Proc sini( Pchar s, short i);
Proc sfix(Pchar s, short i, short max);
Func short maxlen(Pchar s);
Func Pchar pscopy( Pchar s, Pchar a, short i,short j);
Func Bool scopy( Pchar a, Pchar b);
Func Bool ccopy( Pchar a, char c);
Func Bool sadd( Pchar s, Pchar t);
Func Bool nadd( Pchar s, long n);
Func Bool cadd( Pchar s, char c);
Func Bool sins( Pchar s, Pchar t);
Func Bool cins( Pchar s, char c);
Func short cpos( char c, Pchar s);
Func short spos( Pchar sub, Pchar s);

Func short length(Pchar s);
Func Bool steq(Pchar s, Pchar t);
Func Bool stne(Pchar s, Pchar t);
Func short scompare(Pchar a, Pchar b);
Func short ord(char c);
Func short pred(short i);
Func short succ(short i);
Proc stri(long n, Pchar s);
Proc strif(long n, short f, Pchar s);
Proc strf(double x, short a, short b, Pchar s); /* float -> string */
Func long   ival(Pchar s, short *err);
Func double rval(Pchar s, short *err);

Func char upcase(char c);
Func char lowcase(char c);
Func short hi(long w);
Func short lo(long w);
Func Bool odd(long x);
Func Bool alfa(char c);
Func Bool num(char c);
Func Bool alfanum(char c);
Func Pchar stupcase( Pchar s);

/***** primitive input-output ***/
Proc wc(char c);
Proc wln(void);
Proc ws( Pchar s);
Proc wi(long i);
Proc rs( Pchar s);
Func char rc(void);

Func short freadstr(Pfile f, Pchar s, short max);
Func char freadc(Pfile f);
Func long freadi(Pfile f);

Func long np_round(double d);	// sjb to avoid clash with round() in math.h
Func long np_trunc(double x);	// sjb to avoid clash with trunc() in math.h
Func double sqr(double x);
Func double absf(double x); /* abs */
Func long absi( long i);
Func double frac(double x);

Func Bool reset(Pfile f);
Func Bool rewrite(Pfile f);
Proc rawcopy(Pointer a, Pointer b, short la, short lb);
Func Pointer new(long sz);
Proc dispose(Pointer p);
Func Pchar newstring(short n);

