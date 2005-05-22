/*       mystring.c                Copyright (C)  2002    Georg Post
 *
 *  This file is part of Numparam, see:  readme.txt  
 *  Free software under the terms of the GNU Lesser General Public License
 * $Id$
 */

#ifdef __TURBOC__
extern unsigned _stklen= 32000; /* Turbo C default was only 4 K */
#endif

#include <stdio.h>
#include <stdlib.h>
/* #include <math.h>  -- ceil floor */
#include "config.h"
#ifdef HAS_WINDOWS
#include "wstdio.h"
#endif

#include "general.h"

#define Getmax(s,ls)  (((Byte)(s[ls+1])) Shl 8) + (Byte)(s[ls+2])

/***** primitive input-output ***/

Proc wc(char c)
Begin 
  fputc(c, stdout)
EndProc

Proc wln(void)
Begin wc('\n') EndProc

Proc ws( Pchar s)
Begin
  short k=0;
  While s[k] !=0 Do
    wc(s[k]); Inc(k)
  Done
EndProc

Proc wi(long i)
Begin
  Str(16,s);
  nadd(s,i);
  ws(s)
EndProc

Proc rs( Pchar s)
Begin /*basic line input, limit= 80 chars */
  short max,i;
  char c;
  max=maxlen(s); 
  i=0; sini(s,max);
  If max>80 Then max=80 EndIf
  Repeat
    c=fgetc(stdin);
    If (i<max) And (c>=' ') Then 
      cadd(s,c); Inc(i)
    EndIf
  Until (c==Cr) Or (c=='\n') EndRep
  /* return i */
EndFunc

Func char rc(void)
Begin
  short ls;
  Str(80,s);
  rs(s); ls=length(s);
  If ls>0 Then
    return s[ls-1] 
  Else 
    return 0 
  EndIf
EndProc

/*******  Strings ************
 *  are 0-terminated char arrays with a 2-byte trailer: max length.
 *  the string mini-library is "overflow-safe" under these conditions:
 *    use Str(n,s) macro: define and initialize a string s of maxlen n<255
 *    use sini() to initialize empty strings;  sfix() for non-empty ones.
 *    the Sini() macro does automatic sizing, for automatic char arrays
 *    to allocate a string on the heap, use newstring(n).
 *    use maxlen() and length() to retrieve string max and actual length
 *    use: cadd, cins, sadd, sins, scopy, pscopy to manipulate them
 *    never put '\x0' characters inside strings !
 *
 *    the 'killer idea' is the following:
 *    on string overflow and/or on heap allocation failure, a program
 *    MUST die. 
 */

Intern 
Proc stringbug(Pchar op, Pchar s, Pchar t, char c)
/* we brutally stop the program on string overflow */
Begin
  char rep=' ';
  ws(" STRING overflow "); 
  ws(op); wln();
  ws(" Operand1: "); 
  ws(s); wln();
  If t != Null Then 
    ws(" Operand2: "); 
    ws(t); wln(); 
  EndIf
  If c != 0 Then 
    wc('{'); wc(c); wc('}') 
  EndIf
  ws(" [A]bort [I]gnore ? "); 
  rep=rc();
  If upcase(rep)=='A' Then exit(1) EndIf
EndProc

Proc sini(Pchar s, short max) /* suppose s is allocated */
Begin
  If max<1 Then 
    max=1 
  ElsIf max>Maxstr Then 
    max=Maxstr 
  EndIf
  s[0]=0; 
  s[1]= Hi(max); s[2]= Lo(max);
EndProc

Proc sfix(Pchar s, short i, short max)
/* suppose s is allocated and filled with non-zero stuff */
Begin
  short j;
  If max<1 Then 
    max=1 
  ElsIf max>Maxstr Then 
    max=Maxstr 
  EndIf
  If i>max Then
    i=max 
  ElsIf i<0 Then 
    i=0 
  EndIf
  s[i]=0; 
  s[i+1]= Hi(max); s[i+2]= Lo(max);
  For j=0;j<i; Inc(j) Do /* eliminate null characters ! */
    If s[j]==0 Then s[j]=1 EndIf
  Done
EndProc

Intern
Proc inistring(Pchar s, char c, short max)
/* suppose s is allocated. empty it if c is zero ! */
Begin
  short i=0;
  s[i]=c;
  If c!=0 Then 
    Inc(i); s[i]=0 
  EndIf
  If max<1 Then 
    max=1 
  ElsIf max>Maxstr Then 
    max=Maxstr 
  EndIf
  s[i+1]= Hi(max); s[i+2]= Lo(max);
EndProc

Func short length(Pchar s)
Begin
  short lg=0;
  While NotZ(s[lg]) Do Inc(lg) Done
  return lg
EndFunc

Func short maxlen(Pchar s)
Begin
  short ls= length(s);
  return Getmax(s,ls)
EndFunc

Func Bool sadd( Pchar s, Pchar t)
Begin
  Bool ok;
  short i=0, max, ls= length(s);
  max= Getmax(s,ls);
  While (t[i] !=0) And (ls<max) Do
    s[ls]= t[i]; 
    Inc(i); Inc(ls);
  Done
  s[ls]=0; 
  s[ls+1]= Hi(max); s[ls+2]= Lo(max);
  ok= (t[i]==0); /* end of t is reached */
  If Not ok Then
    stringbug("sadd",s,t,0) 
  EndIf
  return ok
EndProc

Func Bool cadd( Pchar s, char c)
Begin
  short max, ls= length(s);
  Bool ok;
  max= Getmax(s,ls);
  ok= (ls<max);
  If ok Then
    s[ls+3]= s[ls+2]; s[ls+2]=s[ls+1]; 
    s[ls+1]=0; s[ls]=c
  EndIf
  If Not ok Then
    stringbug("cadd",s, Null,c)
  EndIf
  return ok
EndProc

Func Bool cins( Pchar s, char c)
Begin
  short i, max, ls= length(s);
  Bool ok;
  max= Getmax(s,ls);
  ok= (ls<max);
  If ok Then
    For i=ls+2; i>=0; Dec(i) Do s[i+1]=s[i] Done;
    s[0]=c;
  EndIf
  If Not ok Then
    stringbug("cins",s, Null,c)
  EndIf
  return ok
EndProc

Func Bool sins( Pchar s, Pchar t)
Begin
  short i, max, ls= length(s), lt=length(t);
  Bool ok;
  max= Getmax(s,ls);
  ok= ((ls+lt) < max);
  If ok Then
    For i=ls+2; i>=0; Dec(i) Do s[i+lt]=s[i] Done;
    For i=0; i<lt; Inc(i) Do s[i]=t[i] Done;
  EndIf
  If Not ok Then
    stringbug("sins",s, t,0)
  EndIf
  return ok
EndProc

Func short cpos(char c, Pchar s)
/* return position of c in s, or 0 if not found.
 * BUG, Pascal inherited: first char is at 1, not 0 !
 */
Begin
  short i=0;
  While (s[i] !=c) And (s[i] !=0) Do Inc(i) Done
  If s[i]==c Then
    return (i+1)
  Else
    return 0
  EndIf
EndFunc

Func char upcase(char c)
Begin
  If (c>='a')And(c<='z') Then
    return c+'A'-'a'
  Else
    return c
  EndIf
EndFunc

Func Bool scopy(Pchar s, Pchar t) /* returns success flag */
Begin
  Bool ok;
  short i,max, ls= length(s);
  max= Getmax(s,ls);
  i=0;
  While (t[i] !=0) And (i<max) Do
    s[i]= t[i]; Inc(i);
  Done
  s[i]=0; 
  s[i+1]= Hi(max); s[i+2]= Lo(max);
  ok= (t[i]==0); /* end of t is reached */
  If Not ok Then
    stringbug("scopy",s, t,0)
  EndIf
  return ok
EndProc

Func Bool ccopy(Pchar s, char c) /* returns success flag */
Begin
  short max, ls= length(s);
  Bool ok=False;
  max= Getmax(s,ls);
  If max>0 Then
    s[0]=c; sfix(s,1,max);
    ok=True
  EndIf
  If Not ok Then
    stringbug("ccopy",s, Null,c)
  EndIf
  return ok
EndProc

Func Pchar pscopy(Pchar s, Pchar t, short start, short leng)
/* partial string copy, with Turbo Pascal convention for "start" */
/* BUG: position count starts at 1, not 0 ! */
Begin
  short max= maxlen(s); /* keep it for later */
  short stop= length(t);
  short i;
  Bool ok= (max>=0) And (max<=Maxstr);
  If Not ok Then
    stringbug("copy target non-init", s, t, 0)
  EndIf 
  If leng>max Then
    leng=max; ok=False
  EndIf
  If start>stop Then /* nothing! */
    ok=False; 
    inistring(s,0,max)
  Else
    If (start+leng-1)>stop Then
      leng = stop-start+1; 
      ok=False
    EndIf
    For i=0; i<leng; Inc(i) Do s[i]= t[start+i -1] Done
    i=leng; s[i]=0;  
    s[i+1]= Hi(max); s[i+2]= Lo(max);
  EndIf
  /* If Not ok Then stringbug("copy",s, t, 0) EndIf */
  /* If ok Then return s Else return Null EndIf */
  ok=ok;
  return s
EndProc

Func short ord(char c)
Begin
  return c AND 0xff
EndFunc /* strip high byte */

Func short pred(short i)
Begin
 return (--i)
EndFunc

Func short succ(short i)
Begin
  return (++i)
EndFunc

Func Bool nadd( Pchar s, long n)
/* append a decimal integer to a string */
Begin
  short d[25];
  short j,k,ls,len;
  char sg;  /* the sign */
  Bool ok;
  k=0;
  len=maxlen(s);
  If n<0 Then
    n= -n; sg='-' 
  Else
    sg='+'
  EndIf
  While n>0 Do 
    d[k]=n Mod 10; Inc(k); 
    n= n Div 10
  Done
  If k==0 Then 
    ok=cadd(s,'0')
  Else
    ls=length(s);
    ok= (len-ls)>k;
    If ok Then
      If sg=='-' Then
        s[ls]=sg; Inc(ls)
      EndIf
      For j=k-1; j>=0; Dec(j) Do
        s[ls]=d[j]+'0'; Inc(ls)
      Done
      sfix(s,ls,len);
    EndIf
  EndIf
  If Not ok Then
    stringbug("nadd",s, Null,sg)
  EndIf
  return ok
EndProc

Proc stri( long n, Pchar s)
/* convert integer to string */
Begin
  sini(s, maxlen(s));
  nadd(s,n)
EndProc

Proc rawcopy(Pointer a, Pointer b, short la, short lb)
/* dirty binary copy */
Begin
  short j,n;
  If lb<la Then
    n=lb 
  Else
    n=la
  EndIf
  For j=0; j<n; Inc(j) Do
    ((Pchar)a)[j]=((Pchar)b)[j]
  Done
EndProc

Func short scompare(Pchar a, Pchar b)
Begin
  Word j=0; 
  short k=0;
  While (a[j]==b[j]) And (a[j]!=0) And (b[j]!=0) Do Inc(j) Done;
  If a[j]<b[j] Then
    k= -1 
  ElsIf a[j]>b[j] Then
    k=1
  EndIf
  return k
EndFunc

Func Bool steq(Pchar a, Pchar b) /* string a==b test */
Begin
  Word j=0;
  While (a[j]==b[j]) And (a[j]!=0) And (b[j]!=0) Do Inc(j) Done;
  return ((a[j]==0) And (b[j]==0)) /* string equality test */
EndFunc

Func Bool stne(Pchar s, Pchar t)
Begin
  return scompare(s,t) !=0
EndFunc

Func short hi(long w)
Begin
  return (w AND 0xff00) Shr 8
EndFunc

Func short lo(long w)
Begin
  return (w AND 0xff)
EndFunc

Func char lowcase(char c)
Begin
  If (c>='A')And(c<='Z') Then
    return (char)(c-'A' +'a')
  Else
    return c
  EndIf
EndFunc

Func Bool alfa( char  c)
Begin
  return ((c>='a') And (c<='z')) Or ((c>='A') And (c<='Z'));
EndFunc

Func  Bool num( char  c)
Begin
  return (c>='0') And (c<='9');
EndFunc

Func Bool alfanum(char c)
Begin 
  return
  ((c>='a') And (c<='z')) Or ((c>='A')And(c<='Z')) 
  Or ((c>='0')And(c<='9'))
  Or (c=='_')
EndFunc

Func short freadstr(Pfile f, Pchar s, short max)
/* read a line from a file. 
   BUG: long lines truncated without warning, ctrl chars are dumped.
*/
Begin 
  char c; 
  short i=0, mxlen=maxlen(s); 
  If mxlen<max Then max=mxlen EndIf
  Repeat 
     c=fgetc(f); /*  tab is the only control char accepted */
    If ((c>=' ') Or (c<0) Or (c==Tab)) And (i<max) Then
      s[i]=c; Inc(i)
    EndIf
  Until feof(f) Or (c=='\n') EndRep
  s[i]=0; 
  s[i+1]= Hi(mxlen); s[i+2]= Lo(mxlen);
  return i
EndProc

Func char freadc(Pfile f)
Begin
  return fgetc(f)
EndFunc

Func long freadi(Pfile f)
/* reads next integer, but returns 0 if none found. */
Begin 
  long z=0;
  Bool minus=False;
  char c;
  Repeat c=fgetc(f) 
  Until feof(f) Or  Not ((c>0) And (c<=' ')) EndRep /* skip space */
  If c=='-' Then
    minus=True; c=fgetc(f)
  EndIf
  While num(c) Do
    z= 10*z + c-'0'; c=fgetc(f)
  Done
  ungetc(c,f) ; /* re-push character lookahead */
  If minus Then z= -z EndIf;
  return z
EndFunc

Func Pchar stupcase( Pchar s)
Begin
  short i=0; 
  While s[i] !=0 Do
    s[i]= upcase(s[i]); Inc(i)
  Done
  return s
EndFunc

/*****  pointer tricks: app won't use naked malloc(), free() ****/

Proc dispose(Pointer p)
Begin 
  If p != Null Then free(p) EndIf 
EndProc

Func Pointer new(long sz)
Begin
  Pointer p;
  If sz<=0 Then
    return Null
  Else
#ifdef __TURBOC__ 
    /* truncate to 64 K ! */
    If sz> 0xffff Then sz= 0xffff EndIf
    p= malloc((Word)sz);
#else
    p= malloc(sz);
#endif
    If p==Null Then /* fatal error */
      ws(" new() failure. Program halted.\n");
      exit(1);
    EndIf
    return p
  EndIf
EndFunc

Func Pchar newstring(short n)
Begin
  Pchar s= (Pchar)new(n+4);
  sini(s, n); 
  return s
EndFunc

/***** elementary math *******/

Func double sqr(double x)
Begin
  return x*x
EndFunc

Func double absf(double x)
Begin
  If x<0.0 Then
    return -x
  Else
    return x
  EndIf
EndFunc

Func long  absi(long i)
Begin
  If i>=0 Then
    return(i)
  Else
    return(-i)
  EndIf
EndFunc

Proc strif(long i, short f, Pchar s)
/* formatting like str(i:f,s) in Turbo Pascal */
Begin
  short j,k,n,max;
  char cs;
  char t[32];
  k=0;
  max=maxlen(s);
  If i<0 Then
    i= -i; cs='-'
  Else
    cs=' '
  EndIf;
  While i>0 Do
    j=(short)(i Mod 10);
    i=(long)(i Div 10);
    t[k]=chr('0'+j); Inc(k)
  Done
  If k==0 Then
    t[k]='0'; Inc(k)
  EndIf
  If cs=='-' Then
    t[k]=cs
  Else
    Dec(k)
  EndIf;
    /* now the string  is in 0...k in reverse order */
  For j=1; j<=k; Inc(j) Do t[k+j]=t[k-j] Done /* mirror image */
  t[2*k+1]=0; /* null termination */
  n=0;
  If (f>k) And (f<40) Then /* reasonable format */
    For j=k+2; j<=f; Inc(j) Do
      s[n]=' '; Inc(n)
    Done
  EndIf
  For j=0; j<=k+1; Inc(j) Do s[n+j]=t[k+j] Done; /* shift t down */
  k=length(s); 
  sfix(s,k,max);
EndProc

Func Bool odd(long x)
Begin
  return NotZ(x AND 1)
EndFunc

Func short vali(Pchar s, long * i)
/* convert s to integer i. returns error code 0 if Ok */
/* BUG: almost identical to ival() with arg/return value swapped ... */
Begin
  short k=0, digit=0, ls;
  long z=0;
  Bool minus=False, ok=True;
  char c;
  ls=length(s);
  Repeat 
    c=s[k]; Inc(k) 
  Until (k>=ls) Or  Not ((c>0) And (c<=' ')) EndRep /* skip space */
  If c=='-' Then
    minus=True; 
    c=s[k]; Inc(k)
  EndIf
  While num(c) Do
    z= 10*z + c-'0';
    c=s[k]; Inc(k);
    Inc(digit)
  Done
  If minus Then z= -z EndIf;
  *i= z; 
  ok= (digit>0) And (c==0); /* successful end of string */
  If ok Then
    return 0
  Else
    return k /* one beyond error position */
  EndIf
EndFunc

Intern 
Func Bool match
  (Pchar s, Pchar t, short n, short tstart, Bool testcase)
Begin
/* returns 0 If tstart is out of range. But n may be 0 ? */
/* True if s matches t[tstart...tstart+n]  */
  short i,j,lt;
  Bool ok;
  char a,b;
  i=0; j=tstart;
  lt= length(t);
  ok=(tstart<lt);
  While ok And (i<n) Do
    a=s[i]; b=t[j];
    If Not testcase Then
      a=upcase(a); b=upcase(b)
    EndIf
    ok= (j<lt) And (a==b);
    Inc(i); Inc(j);
  Done
  return ok
EndFunc

Intern 
Func short posi(Pchar sub, Pchar s, short opt)
/* find position of substring in s */
Begin
  /* opt=0: like Turbo Pascal */
 /*  opt=1: like Turbo Pascal Pos, but case insensitive */
 /*  opt=2: position in space separated wordlist for scanners */
  short a,b,k,j;
  Bool ok, tstcase;
  Str(250,t);
  ok=False;
  tstcase=( opt==0);
  If opt<=1 Then
    scopy(t,sub)
  Else
    cadd(t,' '); sadd(t,sub); cadd(t,' ');
  EndIf
  a= length(t); 
  b= (short)(length(s)-a);
  k=0; j=1;
  If a>0 Then  /*Else return 0*/
    While (k<=b) And (Not ok) Do
      ok=match(t,s, a,k, tstcase); /* we must start at k=0 ! */
      Inc(k);
      If s[k]==' ' Then Inc(j) EndIf /* word counter */
    Done
  EndIf
  If opt==2 Then k=j EndIf
  If ok Then
    return k
  Else
    return 0
  EndIf
EndFunc

Func short spos(Pchar sub, Pchar s)
/* equivalent to Turbo Pascal pos().
   BUG: counts 1 ... length(s), not from 0 like C  
*/
Begin
  return posi( sub, s, 0)
EndFunc

/**** float formatting with printf/scanf ******/

Func short valr(Pchar s, double *r)
/* returns 0 if ok, else length of partial string ? */
Begin
  short n=sscanf(s, "%lG", r);
  If n==1 Then
    return(0)
  Else
    return(1)
  EndIf
EndFunc

Proc strf( double x, short f1, short f2, Pchar t)
/* e-format if f2<0, else f2 digits after the point, total width=f1 */
/* if f1=0, also e-format with f2 digits */
Begin /*default f1=17, f2=-1*/
  Str(30,fmt);
  short n,mlt;
  mlt=maxlen(t);
  cadd(fmt,'%');
  If f1>0 Then
    nadd(fmt , f1); /* f1 is the total width */
    If f2<0 Then
      sadd(fmt,"lE") /* exponent format */
    Else
      cadd(fmt,'.');
      nadd(fmt,f2);
      sadd(fmt,"lf")
    EndIf
  Else
    cadd(fmt,'.');
    nadd(fmt, absi(f2-6)); /* note the 6 surplus positions */
    cadd(fmt,'e');
  EndIf
  n=sprintf(t, fmt, x);
  sfix(t,n, mlt);
EndProc

Func double rval(Pchar s, short *err)
/* returns err=0 if ok, else length of partial string ? */
Begin
  double r= 0.0;
  short n=sscanf(s, "%lG", &r);
  If n==1 Then
    (*err)=0
  Else
    (*err)=1
  EndIf
  return r;
EndFunc

Func long ival(Pchar s, short *err) 
/* value of s as integer string.  error code err= 0 if Ok */
Begin 
  short k=0, digit=0, ls; 
  long z=0;
  Bool minus=False, ok=True;
  char c;
  ls=length(s);
  Repeat
    c=s[k]; Inc(k) 
  Until (k>=ls) Or  Not ((c>0) And (c<=' ')) EndRep /* skip space */
  If c=='-' Then
    minus=True;
    c=s[k]; Inc(k)
  EndIf
  While num(c) Do
    z= 10*z + c-'0';
    c=s[k]; Inc(k);
    Inc(digit)
  Done
  If minus Then z= -z EndIf;
  ok= (digit>0) And (c==0); /* successful end of string */
  If ok Then
    (*err)= 0
  Else
    (*err)= k /* one beyond error position */
  EndIf
  return z
EndFunc

#ifndef _MATH_H

Func long np_round(double x)
/* using <math.h>, it would be simpler: floor(x+0.5) */
Begin
  double u; 
  long z; 
  short n;
  Str(40,s);
  u=2e9; 
  If x>u Then
    x=u
  ElsIf x< -u Then
    x= -u
  EndIf
  n=sprintf(s,"%-12.0f", x);
  s[n]=0; 
  sscanf(s,"%ld", Addr(z));
  return z
EndFunc

Func long np_trunc(double x)
Begin
  long n=np_round(x);
  If (n>x) And (x>=0.0) Then
    Dec(n) 
  ElsIf (n<x) And (x<0.0) Then
    Inc(n)
  EndIf
  return n
EndFunc

Func double frac(double x)
Begin
  return x- np_trunc(x) 
EndFunc

Func double intp(double x)
Begin
  double u=2e9;
  If (x>u) Or (x< -u) Then
    return x
  Else
    return np_trunc(x)
  EndIf
EndFunc

#else  /* use floor() and ceil() */

Func long np_round(double r)
Begin
  return (long)floor(r+0.5)
EndFunc

Func long np_trunc(double r)
Begin
  If r>=0.0 Then
    return (long)floor(r)
  Else
    return (long)ceil(r)
  EndIf
EndFunc

Func double frac(double x)
Begin
  If x>=0.0 Then
    return(x - floor(x))
  Else
    return(x - ceil(x))
  EndIf
EndFunc

Func double intp(double x) /* integral part */
Begin
  If x>=0.0 Then
    return floor(x)
  Else
    return ceil(x)
  EndIf
EndFunc

#endif  /* _MATH_H */


