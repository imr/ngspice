/*       xpressn.c                Copyright (C)  2002    Georg Post
 *
 *  This file is part of Numparam, see:  readme.txt  
 *  Free software under the terms of the GNU Lesser General Public License 
 */

#include <stdio.h>   /* for function message() only. */
#include <math.h>

#include "general.h"
#include "numparam.h"

/************ keywords ************/

/* SJB - 150 chars is ample for this - see initkeys() */
Intern Str(150, keys); /*all my keywords*/
Intern Str(150, fmath); /* all math functions */
 
Intern
Proc initkeys(void)
/* the list of reserved words */
Begin
  scopy(keys,
  "and or not div mod if else end while macro funct defined"
  " include for to downto is var"); 
  stupcase(keys);
  scopy(fmath, "sqr sqrt sin cos exp ln arctan abs pwr"); 
  stupcase(fmath);
EndProc

Intern
Func double mathfunction(short f, double z, double x)
/* the list of built-in functions. Patch 'fmath' and here to get more ...*/
Begin
  double y;
  Switch f
  CaseOne 1 Is  y= x*x
  Case    2 Is  y= sqrt(x)
  Case    3 Is  y= sin(x)
  Case    4 Is  y= cos(x)
  Case    5 Is  y= exp(x)
  Case    6 Is  y= ln(x)
  Case    7 Is  y= atan(x)
  Case    8 Is  y= fabs(x)
  Case    9 Is  y= exp( x* ln(fabs(z))) 
     /* pwr(,): the only one with 2 args */
  Default y=x EndSw
  return y
EndFunc

Cconst(Defd,12)
/* serial numb. of 'defined' keyword. The others are not used (yet) */

Intern
Func  Bool message( tdico * dic, Pchar s)
/* record 'dic' should know about source file and line */
Begin
  Strbig(Llen,t);
  Inc( dic->errcount);
  If (dic->srcfile != Null) And NotZ(dic->srcfile[0]) Then  
    scopy(t, dic->srcfile); cadd(t,':')
  EndIf
  If dic->srcline >=0 Then
    nadd(t,dic->srcline); sadd(t,": ");
  EndIf
  sadd(t,s); cadd(t,'\n');
  fputs(t,stderr); 
  return True /*error!*/
EndFunc

Proc debugwarn( tdico *d, Pchar s)
Begin
  message(d,s);
  Dec( d->errcount)
EndProc

/************* historical: stubs for nodetable manager ************/
/* in the full preprocessor version there was a node translator for spice2 */

Intern
Proc initsymbols(auxtable * n)
Begin
EndProc

Intern
Proc donesymbols(auxtable * n)
Begin
EndProc

/* Intern
Func short parsenode(auxtable *n, Pchar s)
Begin
  return 0
EndFunc
*/

/************ the input text symbol table (dictionary) *************/

Proc initdico(tdico * dico)
Begin
  short i;
  dico->nbd=0;  
  Sini(dico->option);
  Sini(dico->srcfile);
  dico->srcline= -1;
  dico->errcount= 0;
  For i=0; i<=Maxdico; Inc(i) Do
    sini(dico->dat[i].nom,20) 
  Done
  dico->tos= 0; 
  dico->stack[dico->tos]= 0; /* global data beneath */
  initsymbols(Addr(dico->nodetab));
  initkeys();
EndProc

/*  local semantics for parameters inside a subckt */
/*  arguments as wll as .param expressions  */  
/* to do:  scope semantics ?
   "params:" and all new symbols should have local scope inside subcircuits.
   redefinition of old symbols gives a warning message.
*/

Cconst(Push,'u')
Cconst(Pop, 'o')

Intern
Proc dicostack(tdico *dico, char op) 
/* push or pop operation for nested subcircuit locals */
Begin

  If op==Push Then
    If dico->tos < (20-1) Then Inc(dico->tos)
    Else message(dico, " Subckt Stack overflow")
    EndIf 
    dico->stack [dico->tos]= dico->nbd;
  ElsIf op==Pop Then
    /*       obsolete:  undefine all data items of level dico->tos  
    For i=dico->nbd; i>0; Dec(i) Do 
      c= dico->dat[i].tp;
      If ((c=='R') Or (c=='S')) And (dico->dat[i].level == dico->tos) Then 
        dico->dat[i].tp= '?' 
      EndIf 
    Done
    */
    If dico->tos >0 Then
      dico->nbd= dico->stack[dico->tos]; /* simply kill all local items */
      Dec(dico->tos)
    Else message(dico," Subckt Stack underflow.") 
    EndIf 
  EndIf
EndProc

Func short donedico(tdico * dico)
Begin
  short sze= dico->nbd;
  donesymbols(Addr(dico->nodetab));
  return sze;
EndProc

Intern
Func  short entrynb( tdico * d, Pchar s)
/* symbol lookup from end to start,  for stacked local symbols .*/
/* bug: sometimes we need access to same-name symbol, at lower level? */
Begin
  short i;
  Bool ok;
  ok=False;
  i=d->nbd+1;
  While (Not ok) And (i>1) Do
    Dec(i);
    ok= steq(d->dat[i].nom, s);
  Done
  If Not ok Then 
    return 0 
  Else 
    return i 
  EndIf
EndFunc

Func char getidtype( tdico *d, Pchar s)
/* test if identifier s is known. Answer its type, or '?' if not in list */
Begin
  char itp='?'; /* assume unknown */
  short i= entrynb(d, s);
  If i >0 Then itp= d->dat[i].tp EndIf
  return itp
EndFunc

Intern
Func double fetchnumentry(
 tdico * dico,
 Pchar t,
 Bool * perr)
Begin
  Bool err= *perr;
  Word k;
  double u;
  Strbig(Llen, s);
  k=entrynb(dico,t); /*no keyword*/
  /*dbg -- If k<=0 Then ws("Dico num lookup fails. ") EndIf */
  While (k>0) And (dico->dat[k].tp=='P') Do 
    k= dico->dat[k].ivl 
  Done  /*pointer chain*/
  If k>0 Then
    If dico->dat[k].tp!='R' Then k=0 EndIf
  EndIf
  If k>0 Then 
    u=dico->dat[k].vl 
  Else
    u=0.0; 
    scopy(s,"Undefined number ["); sadd(s,t); cadd(s,']');
    err=message( dico, s)
  EndIf
  *perr= err;
  return u
EndFunc

/*******  writing dictionary entries *********/

Intern
Func  short attrib( tdico * dico, Pchar t, char  op)
Begin
/* seek or attribute dico entry number for string t.
   Option  op='N' : force a new entry, if tos>level and old is  valid.
*/
  short i;
  Bool ok;
  i=dico->nbd+1;
  ok=False;
  While (Not ok) And (i>1) Do /*search old*/
    Dec(i); 
    ok= steq(dico->dat[i].nom,t);
  Done
  If ok And (op=='N') 
     And ( dico->dat[i].level < dico->tos)
     And ( dico->dat[i].tp != '?')
  Then ok=False EndIf
  If Not ok Then
    Inc(dico->nbd); 
    i= dico->nbd;
    If dico->nbd > Maxdico Then 
      i=0 
    Else
      scopy(dico->dat[i].nom,t); 
      dico->dat[i].tp='?'; /*signal Unknown*/
      dico->dat[i].level= dico->tos;
    EndIf
  EndIf
  return i
EndFunc

Intern
Func  Bool define(
 tdico * dico,
 Pchar t,      /* identifier to define */
 char  op,     /* option */
 char tpe,     /* type marker */
 double  z,    /* float value if any */
 Word  w,      /* integer value if any */ 
 Pchar base)   /* string pointer if any */
Begin
/*define t as real or integer, 
  opcode= 'N' impose a new item under local conditions. 
  check for pointers, too, in full macrolanguage version:
     Call with 'N','P',0.0, ksymbol ... for VAR parameter passing.
  Overwrite warning, beware: During 1st pass (macro definition),
      we already make symbol entries which are dummy globals !
      we mark each id with its subckt level, and warn if write at higher one.
*/
  short i;
  char c;
  Bool err, warn;
  Strbig(Llen,v);
  i=attrib(dico,t,op); 
  err=False;
  If i<=0 Then 
    err=message( dico," Symbol table overflow")
  Else
    If dico->dat[i].tp=='P' Then 
      i= dico->dat[i].ivl 
    EndIf; /*pointer indirection*/
    If i>0 Then 
      c=dico->dat[i].tp 
    Else 
      c=' ' 
    EndIf
    If (c=='R') Or (c=='S') Or (c=='?') Then
      dico->dat[i].vl=z; 
      dico->dat[i].tp=tpe; 
      dico->dat[i].ivl=w;
      dico->dat[i].sbbase= base;
      /* If (c !='?') And (i<= dico->stack[dico->tos]) Then  */
      If c=='?' Then 
        dico->dat[i].level= dico->tos 
      EndIf /* promote! */ 
      If dico->dat[i].level < dico->tos Then 
        /* warn about re-write to a global scope! */
        scopy(v,t); cadd(v,':'); 
        nadd(v,dico->dat[i].level);
        sadd(v," overwritten.");  
        warn=message( dico,v);
      EndIf 
    Else
      scopy(v,t); 
      sadd(v,": cannot redefine");  
      err=message( dico,v);
    EndIf
  EndIf
  return err;
EndFunc

Func Bool defsubckt(tdico *dico, Pchar s, Word w, char categ)
/* called on 1st pass of spice source code, 
   to enter subcircuit (categ=U) and model (categ=O) names 
*/
Begin
  Str(80,u);
  Bool err;
  short i,j,ls;
  ls=length(s);
  i=0; 
  While (i<ls) And (s[i] !='.') Do Inc(i) Done /* skip 1st dotword */
  While (i<ls) And (s[i]>' ') Do Inc(i) Done
  While (i<ls) And (s[i]<=' ') Do Inc(i) Done /* skip blank */
  j=i; 
  While (j<ls) And (s[j]>' ') Do Inc(j) Done
  If (j>i) And alfa(s[i]) Then
    pscopy(u,s, i+1, j-i);
    stupcase(u);
    err= define( dico, u, ' ',categ, 0.0, w, Null);
  Else
    err= message( dico,"Subcircuit or Model without name.");
  EndIf
  return err
EndFunc

Func short findsubckt( tdico *dico, Pchar s, Pchar subname)
/* input: s is a subcircuit invocation line.
   returns 0 if not found, else the stored definition line number value
   and the name in string subname  */
Begin
  Str(80,u); /* u= subckt name is last token in string s */
  short i,j,k;
  k=length(s); 
  While (k>=0) And (s[k]<=' ') Do Dec(k) Done
  j=k;
  While (k>=0) And (s[k]>' ') Do Dec(k) Done
  pscopy(u,s, k+2, j-k);
  stupcase(u);
  i= entrynb(dico,u);
  If (i>0) And (dico->dat[i].tp == 'U')  Then 
    i= dico->dat[i].ivl;
    scopy(subname,u) 
  Else 
    i= 0;
    scopy(subname,"");
    message(dico, "Cannot find subcircuit.");  
  EndIf
  return i
EndFunc  

#if 0   /* unused, from the full macro language... */
Intern
Func  short deffuma(  /* define function or macro entry. */
 tdico * dico, Pchar t, char  tpe, Word bufstart,
 Bool * pjumped, Bool * perr)
Begin
  Bool jumped= *pjumped; Bool err= *perr;
/* if not jumped, define new function or macro, returns index to buffferstart
   if jumped, return index to existing function
*/
  short i,j;
  Strbig(Llen, v);
  i=attrib(dico,t,' '); j=0;
  If i<=0 Then
    err=message( dico," Symbol table overflow")
  Else
    If dico->dat[i].tp != '?' Then /*old item!*/
      If jumped Then 
        j=dico->dat[i].ivl
      Else
        scopy(v,t); sadd(v," already defined");  
        err=message( dico,v)
      EndIf
    Else
      dico->dat[i].tp=tpe; 
      Inc(dico->nfms); j=dico->nfms; 
      dico->dat[i].ivl=j;
      dico->fms[j].start= bufstart; /* =ibf->bufaddr = start addr in buffer */
    EndIf
  EndIf
  *pjumped= jumped; 
  *perr= err;
  return j;
EndFunc
#endif

/************ input scanner stuff **************/

Intern
Func  Byte keyword( Pchar keys, Pchar t)
Begin
/* return 0 if t not found in list keys, else the ordinal number */
 Byte i,j,k;
 short lt,lk;
 Bool ok;
  lt=length(t); 
  lk=length(keys); 
  k=0; j=0;
  Repeat 
    Inc(j); 
    i=0; ok=True;
    Repeat 
      Inc(i); Inc(k); 
      ok= (k<=lk) And (t[i-1]==keys[k-1]);
    Until (Not ok) Or (i>=lt) EndRep
    If ok Then 
      ok=(k==lk) Or (keys[k]<=' ') 
    EndIf
    If Not ok And (k<lk) Then /*skip to next item*/
      While (k<=lk) And (keys[k-1]>' ') Do Inc(k) Done
    EndIf
  Until ok Or (k>=lk) EndRep
  If ok Then 
    return j 
  Else 
    return 0 
  EndIf
EndFunc

Intern
Func  double parseunit( double x, Pchar s)
/* the Spice suffixes */
Begin
  double u;
  Str(20, t);
  Bool isunit;
  isunit=True; 
  pscopy(t,s,1,3);
  If steq(t,"MEG") Then
    u=1e6
  ElsIf s[0]=='G' Then
    u=1e9
  ElsIf s[0]=='K' Then
    u=1e3
  ElsIf s[0]=='M' Then
    u=0.001
  ElsIf s[0]=='U' Then
    u=1e-6
  ElsIf s[0]=='N' Then
    u=1e-9
  ElsIf s[0]=='P' Then
    u=1e-12
  ElsIf s[0]=='F' Then
    u=1e-15
  Else 
    isunit=False 
  EndIf
  If isunit Then x=x*u EndIf
  return x
EndFunc

Intern
Func  short fetchid(
 Pchar s, Pchar t,
 short  ls, short i)
/* copy next identifier from s into t, advance and return scan index i */
Begin
  char c;
  Bool ok;
  c=s[i-1];
  While (Not alfa(c)) And (i<ls) Do 
    Inc(i); c=s[i-1] 
  Done
  scopy(t,""); 
  cadd(t,upcase(c)); 
  Repeat 
    Inc(i);
    If i<=ls Then 
      c=s[i-1] 
    Else 
      c=Nul 
    EndIf
    c= upcase(c);
    ok= ((c>='0') And (c<='9')) Or ((c>='A') And (c<='Z'));
    If ok Then cadd(t,c) EndIf
  Until Not ok EndRep
  return i /*return updated i */
EndFunc

Intern
Func  double exists(
 tdico * d,
 Pchar  s,
 short * pi,
 Bool * perror)
/* check if s in smboltable 'defined': expect (ident) and return 0 or 1 */
Begin
  Bool error= *perror; 
  short i= *pi;
  double x;
  short ls;
  char c;
  Bool ok;
  Strbig(Llen, t);
  ls=length(s); 
  x=0.0;
  Repeat 
    Inc(i);
    If i>ls Then
      c=Nul 
    Else 
      c=s[i-1]
    EndIf; 
    ok= (c=='(')
  Until ok Or (c==Nul) EndRep
  If ok Then 
    i=fetchid(s,t, ls,i); Dec(i);
    If entrynb(d,t)>0 Then x=1.0 EndIf
    Repeat 
      Inc(i);
      If i>ls Then 
        c=Nul 
      Else 
        c=s[i-1] 
      EndIf 
      ok= (c==')')
    Until ok Or (c==Nul) EndRep
  EndIf
  If Not ok Then
    error=message( d," Defined() syntax");
  EndIf /*keep pointer on last closing ")" */
  *perror= error; 
  *pi=i;
  return x;
EndFunc

Intern
Func double fetchnumber( tdico *dico,
 Pchar s, short  ls,
 short * pi,
 Bool * perror)
/* parse a Spice number in string s */
Begin
  Bool error= *perror; 
  short i= *pi;
  short k,err;
  char d;
  Str(20, t); 
  Strbig(Llen, v);
  double u;
  k=i;
  Repeat 
    Inc(k);
    If k>ls Then 
      d=chr(0) 
    Else 
      d=s[k-1] 
    EndIf
  Until Not ((d=='.') Or ((d>='0') And (d<='9')))  EndRep
  If (d=='e') Or (d=='E') Then /*exponent follows*/
    Inc(k); d=s[k-1];
    If (d=='+') Or (d=='-') Then Inc(k) EndIf
    Repeat 
      Inc(k); 
      If k>ls Then 
        d=chr(0) 
      Else 
        d=s[k-1] 
      EndIf
    Until Not ((d>='0') And (d<='9')) EndRep
  EndIf
  pscopy(t,s,i, k-i);
  If t[0]=='.' Then 
    cins(t,'0')
  ElsIf t[length(t)-1]=='.' Then 
    cadd(t,'0') 
  EndIf
  u= rval(t, Addr(err));
  If err!=0 Then
    scopy(v,"Number format error: "); 
    sadd(v,t);
    error=message( dico,v)
  Else
    scopy(t,"");
    While alfa(d) Do 
      cadd(t,upcase(d));
      Inc(k); 
      If k>ls Then 
        d=Nul 
      Else 
        d=s[k-1] 
      EndIf
    Done
    u=parseunit(u,t);
  EndIf
  i=k-1;
  *perror= error; 
  *pi=i;
  return u;
EndFunc

Intern
Func  char fetchoperator( tdico *dico,
 Pchar s, short  ls,
 short * pi,
 Byte * pstate, Byte * plevel,
 Bool * perror)
/* grab an operator from string s and advance scan index pi.
   each operator has: one-char alias, precedence level, new interpreter state.
*/
Begin
  short i= *pi; 
  Byte state= *pstate; 
  Byte level= *plevel;
  Bool error= *perror;
  char c,d;
  Strbig(Llen, v);
  c=s[i-1];  
  If i<ls Then 
    d=s[i] 
  Else 
    d=Nul 
  EndIf
  If (c=='!') And (d=='=') Then 
    c='#'; Inc(i)
  ElsIf (c=='<') And (d=='>') Then 
    c='#'; Inc(i)
  ElsIf (c=='<') And (d=='=') Then 
    c='L'; Inc(i)
  ElsIf (c=='>') And (d=='=') Then 
    c='G'; Inc(i)
  ElsIf (c=='*') And (d=='*') Then 
    c='^'; Inc(i)
  ElsIf (c=='=') And (d=='=') Then 
    Inc(i)
  ElsIf (c=='&') And (d=='&') Then 
    Inc(i)
  ElsIf (c=='|') And (d=='|') Then 
    Inc(i)
  EndIf;
  If (c=='+') Or (c=='-') Then
    state=2; /*pending operator*/ 
    level=4;
  ElsIf (c=='*')Or (c=='/') Or (c=='%')Or(c=='\\') Then
    state=2; level=3;
  ElsIf c=='^' Then
    state=2; level=2;
  ElsIf cpos(c,"=<>#GL") >0 Then
    state=2; level= 5;
  ElsIf c=='&' Then
    state=2; level=6;
  ElsIf c=='|' Then
    state=2; level=7;
  ElsIf c=='!' Then
    state=3;
  Else state=0;
    If c>' ' Then
      scopy(v,"Syntax error: letter ["); 
      cadd(v,c); cadd(v,']');
      error=message( dico,v);
    EndIf
  EndIf
  *pi=i; 
  *pstate=state; 
  *plevel=level; 
  *perror=error;
  return c;
EndFunc

Intern
Func  char opfunctkey( tdico *dico,
 Byte  kw, char  c,
 Byte * pstate, Byte * plevel, Bool * perror)
/* handle operator and built-in keywords */
Begin
  Byte state= *pstate; 
  Byte level= *plevel;  
  Bool error= *perror;
/*if kw operator keyword, c=token*/
  Switch kw  /*AND OR NOT DIV MOD  Defined*/
  CaseOne 1 Is 
     c='&'; state=2; level=6
  Case 2 Is 
     c='|'; state=2; level=7
  Case 3 Is
     c='!'; state=3; level=1
  Case 4 Is
     c='\\'; state=2; level=3
  Case 5 Is
     c='%'; state=2; level=3
  Case Defd Is
     c='?'; state=1; level=0
  Default 
     state=0;
     error=message( dico," Unexpected Keyword");
  EndSw /*case*/
  *pstate=state; 
  *plevel=level; 
  *perror=error;
  return c
EndFunc

Intern
Func  double operate(
 char  op,
 double  x,
 double  y)
Begin
/* execute operator op on a pair of reals */
/* bug:   x:=x op y or simply x:=y for empty op?  No error signalling! */
  double u=1.0;
  double z=0.0;
  double epsi=1e-30;
  double t;
  Switch op
  CaseOne ' ' Is  
    x=y; /*problem here: do type conversions ?! */
  Case '+' Is  
    x=x+y;
  Case '-' Is  
    x=x-y;
  Case '*' Is  
    x=x*y;
  Case '/' Is  
    If absf(y)>epsi Then x=x/y EndIf
  Case '^' Is  /*power*/ 
    t=absf(x);
    If t<epsi Then 
      x=z 
    Else 
      x=exp(y*ln(t)) 
    EndIf
  Case '&' Is  /*And*/ 
    If y<x Then x=y EndIf; /*=Min*/
  Case '|' Is  /*Or*/ 
    If y>x Then x=y EndIf;  /*=Max*/
  Case '=' Is  
    If x == y Then x=u Else x=z EndIf;
  Case '#' Is  /*<>*/ 
    If x != y Then x=u Else x=z EndIf;
  Case '>' Is  
    If x>y Then x=u Else x=z EndIf;
  Case '<' Is  
    If x<y Then x=u Else x=z EndIf;
  Case 'G' Is  /*>=*/ 
    If x>=y Then x=u Else x=z EndIf;
  Case 'L' Is  /*<=*/ 
    If x<=y Then x=u Else x=z EndIf;
  Case '!' Is  /*Not*/ 
    If y==z Then x=u Else x=z EndIf;
  Case '%' Is  /*Mod*/ 
    t= np_trunc(x/y); 
    x= x-y*t
  Case '\\' Is  /*Div*/ 
    x= np_trunc(absf(x/y));
  EndSw /*case*/
  return x;
EndFunc

Intern
Func  double formula(
 tdico * dico,
 Pchar  s,
 Bool * perror)
Begin
/* Expression parser. 
  s is a formula with parentheses and math ops +-* / ...
  State machine and an array of accumulators handle operator precedence.
  Parentheses handled by recursion.
  Empty expression is forbidden: must find at least 1 atom.
  Syntax error if no toggle between binoperator And (unop/state1) !
  States : 1=atom, 2=binOp, 3=unOp, 4= stop-codon. 
  Allowed transitions:  1->2->(3,1) and 3->(3,1).
*/
  Cconst(nprece,9) /*maximal nb of precedence levels*/
  Bool error= *perror;
  Byte state,oldstate, topop,ustack, level, kw, fu;
  double u=0.0,v;
  double accu[nprece+1];
  char oper[nprece+1];
  char uop[nprece+1];
  short i,k,ls,natom, arg2;
  char c,d;
  Strbig(Llen, t);
  Bool ok;
  For i=0; i<=nprece; Inc(i) Do 
    accu[i]=0.0; oper[i]=' ' 
  Done
  i=0; 
  ls=length(s);
  While(ls>0) And (s[ls-1]<=' ') Do Dec(ls) Done /*clean s*/
  state=0; natom=0; ustack=0; 
  topop=0; oldstate=0; fu=0; 
  error=False;
  While (i<ls) And (Not error) Do
    Inc(i); c=s[i-1]; 
    If c=='(' Then /*sub-formula or math function */ 
      level=1;
      /* new: must support multi-arg functions */
      k=i; 
      arg2=0; v=1.0;
      Repeat 
        Inc(k);
        If k>ls Then 
          d=chr(0) 
        Else 
          d=s[k-1] 
        EndIf
        If d=='(' Then 
          Inc(level) 
        ElsIf d==')' Then 
          Dec(level) 
        EndIf
        If (d==',') And (level==1) Then arg2=k EndIf /* comma list? */
      Until (k>ls) Or ((d==')') And (level<=0)) EndRep
      If k>ls Then 
        error=message( dico,"Closing \")\" not found.");
        Inc(natom); /*shut up other error message*/
      Else
        If arg2 > i Then
          pscopy(t,s,i+1, arg2-i-1);
	  v=formula( dico, t, Addr(error));
          i=arg2;
        EndIf
        pscopy(t,s,i+1, k-i-1);
	u=formula( dico, t, Addr(error)); 
        state=1; /*atom*/
        If fu>0 Then 
          u= mathfunction(fu,v,u) 
        EndIf
      EndIf
      i=k; fu=0;
    ElsIf alfa(c) Then
      i=fetchid(s,t, ls,i); /*user id, but sort out keywords*/
      state=1; 
      Dec(i);
      kw=keyword(keys,t); /*debug ws('[',kw,']'); */
      If kw==0 Then 
        fu= keyword(fmath,t); /* numeric function? */
        If fu==0 Then
          u=fetchnumentry( dico, t, Addr(error))
        Else 
          state=0 
        EndIf /* state==0 means: ignore for the moment */
      Else 
        c=opfunctkey( dico, kw,c, Addr(state), Addr(level) ,Addr(error)) 
      EndIf
      If kw==Defd Then 
        u=exists( dico, s, Addr(i), Addr(error)) 
      EndIf
    ElsIf ((c=='.') Or ((c>='0') And (c<='9'))) Then
      u=fetchnumber( dico, s,ls, Addr(i), Addr(error));
      state=1;
    Else
      c=fetchoperator(dico, s,ls,
         Addr(i), Addr(state),Addr(level),Addr(error));
      /*may change c to some other operator char!*/
    EndIf /* control chars <' '  ignored*/
    ok= (oldstate==0) Or (state==0) Or
      ((oldstate==1) And (state==2)) Or ((oldstate!=1)And(state!=2));
    If Not ok Then 
      error=message( dico," Misplaced operator") 
    EndIf
    If state==3 Then /*push unary operator*/
      Inc(ustack); 
      uop[ustack]=c;
    ElsIf state==1 Then /*atom pending*/ Inc(natom);
      If i>=ls Then 
        state=4; level=topop 
      EndIf /*close all ops below*/
      For k=ustack; k>=1; Dec(k) Do
        u=operate(uop[k],u,u) 
      Done
      ustack=0;
      accu[0]=u; /* done: all pending unary operators */
    EndIf
    If (state==2) Or (state==4) Then
      /* do pending binaries of priority Upto "level" */
      For k=1; k<=level; Inc(k) Do /* not yet speed optimized! */
        accu[k]=operate(oper[k],accu[k],accu[k-1]);
        accu[k-1]=0.0; 
        oper[k]=' '; /*reset intermediates*/
      Done
      oper[level]=c; 
      If level>topop Then topop=level EndIf
    EndIf
    If (state>0) Then oldstate=state EndIf
  Done /*while*/;
  If (natom==0) Or (oldstate!=4) Then 
    scopy(t," Expression err: "); 
    sadd(t,s);
    error=message( dico,t) 
  EndIf
  *perror= error;
  If error Then
    return 1.0 
  Else 
    return  accu[topop] 
  EndIf
EndFunc /*formula*/

Intern
Func  char fmttype( double  x)
Begin
/* I=integer, P=fixedpoint F=floatpoint*/
/*  find out the "natural" type of format for number x*/
  double ax,dx;
  short rx;
  Bool isint,astronomic;
  ax=absf(x); 
  isint=False; 
  astronomic=False;
  If ax<1e-30 Then
    isint=True;
  ElsIf ax<32000 Then /*detect integers*/ rx=np_round(x);
    dx=(x-rx)/ax; 
    isint=(absf(dx)<1e-6);
  EndIf
  If Not isint Then 
    astronomic= (ax>=1e6) Or (ax<0.01) 
  EndIf
  If isint Then 
    return 'I'
  ElsIf astronomic Then 
    return 'F'
  Else 
    return 'P' 
  EndIf
EndFunc

Intern
Func  Bool evaluate(
 tdico * dico,
 Pchar q,
 Pchar t,
 Byte  mode)
Begin
/* transform t to result q. mode 0: expression, mode 1: simple variable */
  double u=0.0;
  short k,j,lq;
  char dt,fmt;
  Bool numeric, done, nolookup;
  Bool err;
  Strbig(Llen, v);
  scopy(q,""); 
  numeric=False; err=False;
  If mode==1 Then /*string?*/
    stupcase(t);
    k=entrynb(dico,t);
    nolookup= ( k<=0 );
    While (k>0) And (dico->dat[k].tp=='P') Do 
      k=dico->dat[k].ivl 
    Done
      /*pointer chain*/
    If k>0 Then 
      dt=dico->dat[k].tp 
    Else 
      dt=' ' 
    EndIf;
      /*data type: Real or String*/
    If dt=='R' Then
       u=dico->dat[k].vl; numeric=True
    ElsIf dt=='S' Then /*suppose source text "..." at*/
      j=dico->dat[k].ivl; 
      lq=0;
      Repeat 
        Inc(j); Inc(lq); 
        dt= /*ibf->bf[j]; */ dico->dat[k].sbbase[j];
        If cpos('3',dico->option)<=0 Then 
          dt=upcase(dt) 
        EndIf /* spice-2 */
        done= (dt=='\"') Or (dt<' ') Or (lq>99);
        If Not done Then cadd(q,dt) EndIf
      Until done EndRep
    Else k=0 EndIf
    If k <= 0 Then
      scopy(v,""); 
      cadd(v,'\"'); sadd(v,t); 
      sadd(v,"\" not evaluated. ");
      If nolookup Then sadd(v,"Lookup failure.") EndIf
      err=message( dico,v)
    EndIf
  Else 
    u=formula( dico, t, Addr(err)); 
    numeric=True 
  EndIf
  If numeric Then
    fmt= fmttype(u);
    If fmt=='I' Then 
      stri(np_round(u), q) 
    Else 
      strf(u,6,-1,q) 
    EndIf /* strf() arg 2 doesnt work: always >10 significant digits ! */
  EndIf
  return err;
EndFunc

#if 0
Intern
Func  Bool scanline(
 tdico * dico,
 Pchar  s, Pchar r,
 Bool  err)
/* scan host code line s for macro substitution.  r=result line */
Begin
  short i,k,ls,level,nd, nnest;
  Bool spice3;
  char c,d;
  Strbig(Llen, q);
  Strbig(Llen, t);
  Str(20, u);
  spice3= cpos('3', dico->option) >0; /* we had -3 on the command line */
  i=0; ls=length(s); 
  scopy(r,""); 
  err=False; 
  pscopy(u,s,1,3);
  If (ls>7) And steq(u,"**&") Then /*special Comment **&AC #...*/
    pscopy(r,s,1,7); 
    i=7
  EndIf
  While (i<ls) And (Not err) Do
    Inc(i); 
    c=s[i-1];
    If c==Pspice Then /* try pspice expression syntax */
      k=i; nnest=1;
      Repeat 
        Inc(k); d=s[k-1];
        If d=='{' Then 
          Inc(nnest) 
        ElsIf d=='}' Then 
          Dec(nnest) 
        EndIf
      Until (nnest==0) Or (d==0) EndRep
      If d==0 Then 
        err=message( dico,"Closing \"}\" not found.");
      Else
        pscopy(t,s,i+1, k-i-1); 
        err=evaluate( dico, q,t,0);
      EndIf
      i=k;
      If Not err Then /*insert number*/
        sadd(r,q)
      Else 
        err=message( dico,s)
      EndIf
    ElsIf c==Intro Then
      Inc(i);
      While (i<ls) And (s[i-1]<=' ') Do Inc(i) Done
      k=i;
      If s[k-1]=='(' Then /*sub-formula*/ 
        level=1;
        Repeat 
          Inc(k);
          If k>ls Then 
            d=chr(0) 
          Else 
            d=s[k-1] 
          EndIf
          If d=='(' Then 
            Inc(level) 
          ElsIf d==')' Then 
            Dec(level)
          EndIf
        Until (k>ls) Or ((d==')') And (level<=0)) EndRep
	If k>ls Then  
          err=message( dico,"Closing \")\" not found.");
        Else
	  pscopy(t,s,i+1, k-i-1); 
          err=evaluate( dico, q,t,0);
        EndIf
        i=k;
      Else /*simple identifier may also be string*/
        Repeat 
          Inc(k);
          If k>ls Then 
            d=chr(0) 
          Else 
            d=s[k-1] 
          EndIf
        Until (k>ls) Or (d<=' ') EndRep
	pscopy(t,s,i,k-i); 
        err=evaluate( dico, q,t,1);
        i=k-1;
      EndIf
      If Not err Then /*insert the number*/ 
        sadd(r,q)
      Else 
        message( dico,s)
      EndIf
    ElsIf c==Nodekey Then /*follows: a node keyword*/
      Repeat 
        Inc(i) 
      Until s[i-1]>' ' EndRep
      k=i;
      Repeat 
        Inc(k) 
      Until (k>ls) Or Not alfanum(s[k-1]) EndRep
      pscopy(q,s,i,k-i);
      nd=parsenode( Addr(dico->nodetab), q); 
      If Not spice3 Then 
        stri(nd,q) 
      EndIf; /* substitute by number */ 
      sadd(r,q);
      i=k-1;
    Else 
      If Not spice3 Then c=upcase(c) EndIf
      cadd(r,c); /*c<>Intro*/
    EndIf
  Done /*while*/
  return err;
EndFunc
#endif

/********* interface functions for spice3f5 extension ***********/

Intern
Proc compactfloatnb(Pchar v)
/* try to squeeze a floating pt format to 10 characters */ 
/* erase superfluous 000 digit streams before E */
/* bug: truncating, no rounding */ 
Begin
  short n,k, lex;
  Str(20,expo);
  n=cpos('E',v); /* if too long, try to delete digits */
  If n >3 Then
    pscopy(expo, v, n,length(v));
    lex= length(expo);
    k=n-2;  /* mantissa is 0...k */
    While (v[k]=='0') And (v[k-1]=='0') Do Dec(k) Done
    If (k+1+lex) > 10 Then k= 9-lex EndIf
    pscopy(v,v, 1,k+1); 
    sadd(v,expo);   
  EndIf
EndProc

Intern
Func short insertnumber(tdico *dico, short i, Pchar s, Pchar u)
/* insert u in string s in place of the next placeholder number */
Begin
  Str(40,v);
  Str(80,msg);
  Bool found;
  short ls, k; 
  long accu;
  ls= length(s);
  scopy(v,u);
  compactfloatnb(v);
  While length(v)<10 Do 
    cadd(v,' ') 
  Done
  If length(v)>10 Then 
    scopy(msg," insertnumber fails: "); 
    sadd(msg,u); 
    message( dico, msg) 
  EndIf
  found=False;
  While (Not found) And (i<ls) Do
    found= (s[i]=='1');
    k=0; accu=0;
    While found And (k<10) Do /* parse a 10-digit number */  
      found= num(s[i+k]);
      If found Then 
         accu= 10 * accu + s[i+k]- '0' 
      EndIf
      Inc(k)
    Done
    If found Then 
      accu=accu - 1000000000L; /* plausibility test */
      found= (accu>0) And (accu<2000)
    EndIf
    Inc(i)
  Done
  If found Then /* substitute at i-1 */
    Dec(i);
    For k=0; k<10; Inc(k) Do s[i+k]= v[k] Done
    i= i+10;
  Else 
    i= ls; 
    message( dico,"insertnumber: missing slot ");
  EndIf
  return i
EndFunc

Func Bool nupa_substitute( tdico *dico, Pchar s, Pchar r, Bool err)
/* s: pointer to original source line.
   r: pointer to result line, already heavily modified wrt s 
   anywhere we find a 10-char numstring in r, substitute it.
  bug: wont flag overflow!
*/
Begin
  short i,k,ls,level, nnest, ir;
  char c,d;
  Strbig(Llen, q);
  Strbig(Llen, t);
  i=0; 
  ls=length(s); 
  err=False; 
  ir=0;
  While (i<ls) And (Not err) Do
    Inc(i); c=s[i-1];
    If c==Pspice Then /* try pspice expression syntax */
      k=i; nnest=1;
      Repeat 
        Inc(k); d=s[k-1];
        If d=='{' Then 
          Inc(nnest) 
        ElsIf d=='}' Then 
          Dec(nnest) 
        EndIf
      Until (nnest==0) Or (d==0) EndRep
      If d==0 Then 
        err=message( dico,"Closing \"}\" not found.");
      Else
        pscopy(t,s,i+1, k-i-1); 
        err=evaluate( dico, q,t,0);
      EndIf
      i=k;
      If Not err Then
        ir= insertnumber(dico, ir, r,q)
      Else 
        err=message( dico, "Cannot compute substitute")
      EndIf
    ElsIf c==Intro Then
      Inc(i);
      While (i<ls) And (s[i-1]<=' ') Do Inc(i) Done
      k=i;
      If s[k-1]=='(' Then /*sub-formula*/ 
        level=1;
        Repeat 
          Inc(k);
          If k>ls Then 
            d=chr(0) 
          Else 
            d=s[k-1]
          EndIf
          If d=='(' Then 
            Inc(level) 
          ElsIf d==')' Then 
            Dec(level)
          EndIf
        Until (k>ls) Or ((d==')') And (level<=0)) EndRep
	If k>ls Then
          err=message( dico,"Closing \")\" not found.");
        Else
	  pscopy(t,s,i+1, k-i-1);
          err=evaluate( dico, q,t,0);
        EndIf
        i=k;
      Else /*simple identifier may also be string? */
        Repeat 
          Inc(k);
          If k>ls Then
            d=chr(0) 
          Else
            d=s[k-1]
          EndIf
        Until (k>ls) Or (d<=' ') EndRep
	pscopy(t,s,i,k-i); 
        err=evaluate( dico, q,t,1);
        i= k-1;
      EndIf
      If Not err Then
        ir= insertnumber(dico, ir, r,q)
      Else
        message( dico, "Cannot compute &(expression)")
      EndIf
    EndIf
  Done /*while*/
  return err
EndFunc

Intern
Func Byte getword(
 Pchar  s, Pchar t,
 Byte  after,
 short * pi)
/* isolate a word from s after position "after". return i= last read+1 */
Begin
  short i= *pi;
  short ls;
  Byte key;
  i=after;
  ls=length(s);
  Repeat
    Inc(i)
  Until (i>=ls) Or alfa(s[i-1]) EndRep
  scopy(t,"");
  While (i<=ls) And (alfa(s[i-1]) Or num(s[i-1])) Do
    cadd(t,upcase(s[i-1])); 
    Inc(i);
  Done
  If NotZ(t[0]) Then 
     key=keyword(keys,t) 
  Else
     key=0 
  EndIf
  *pi=i;
  return key;
EndFunc

Intern
Func char getexpress( Pchar s, Pchar t, short * pi)
/* returns expression-like string until next separator
 Input  i=position before expr, output  i=just after expr, on separator.
 returns tpe=='R' If numeric, 'S' If string only
*/
Begin
  short i= *pi; 
  short ia,ls,level;
  char c,d, tpe;
  Bool comment= False;
  ls=length(s);
  ia=i+1;
  While (ia<ls) And (s[ia-1]<=' ') Do
    Inc(ia)
  Done /*white space ? */
  If s[ia-1]=='\"' Then /*string constant*/
    Inc(ia); 
    i=ia;
    While (i<ls) And (s[i-1]!='\"') Do Inc(i) Done
    tpe='S';
    Repeat
      Inc(i)
    Until (i>ls) Or (s[i-1] >' ') EndRep
  Else
    If s[ia-1]=='{' Then Inc(ia) EndIf
    i= ia-1;
    Repeat 
      Inc(i); 
      If i>ls Then
        c=';' 
      Else
        c=s[i-1]
      EndIf
      If c=='(' Then /*sub-formula*/ 
        level=1;
        Repeat
          Inc(i);
          If i>ls Then
            d=Nul
          Else
            d=s[i-1]
          EndIf
          If d=='(' Then
            Inc(level)
          ElsIf d==')' Then
            Dec(level)
          EndIf
        Until (i>ls) Or ((d==')') And (level<=0)) EndRep
      EndIf
      /* buggy? */ If (c=='/') Or (c=='-') Then comment= (s[i]==c) EndIf 
    Until (cpos(c, ",;)}") >0)  Or comment EndRep /*legal separators*/
    tpe='R';
  EndIf
  pscopy(t,s,ia,i-ia); 
  If s[i-1]=='}' Then Inc(i) EndIf
  If tpe=='S' Then Inc(i) EndIf /* beyond quote */
  *pi=i; 
  return tpe;
EndFunc

Func Bool nupa_assignment( tdico *dico, Pchar  s, char mode)
/* is called for all 'Param' lines of the input file.
   is also called for the params: section of a subckt .
   mode='N' define new local variable, else global...
   bug: we cannot rely on the transformed line, must re-parse everything!
*/
Begin
/* s has the format: ident = expression; ident= expression ...  */
  Strbig(Llen, t); 
  Strbig(Llen,u);
  short i,j, ls;
  Byte key;
  Bool error, err;
  char dtype;
  Word wval=0;
  double rval= 0.0;
  ls=length(s);
  error=False;
  i=0;
  j= spos("//", s); /* stop before comment if any */
  If j>0 Then ls= j-1 EndIf
   /* bug: doesnt work. need to  revise getexpress ... !!! */
  i=0;
  While (i<ls) And (s[i]<=' ') Do Inc(i) Done
  If s[i]==Intro Then Inc(i) EndIf
  If s[i]=='.' Then  /* skip any dot keyword */
    While s[i]>' ' Do Inc(i) Done
  EndIf
  While (i<ls) And (Not error) Do
    key=getword(s,t, i, Addr(i));
    If (t[0]==0) Or (key>0) Then 
      error=message( dico," Identifier expected")
    EndIf
    If Not error Then /* assignment expressions */
      While (i<=ls) And (s[i-1] !='=') Do Inc(i) Done
      If i>ls Then 
         error= message( dico," = sign expected .") 
      EndIf
      dtype=getexpress(s,u, Addr(i));
      If dtype=='R' Then 
        rval=formula( dico, u, Addr(error));
        If error Then 
          message( dico," Formula() error.") 
        EndIf 
      ElsIf dtype=='S' Then 
        wval= i 
      EndIf
      err=define(dico,t, mode /*was ' ' */ , dtype,rval,wval,Null); 
      error= error Or err;
    EndIf
    If (i<ls) And (s[i-1] != ';') Then 
      error=message( dico," ; sign expected.")
    Else /*Inc(i)*/ 
    EndIf 
  Done
  return error
EndFunc

Func Bool nupa_subcktcall( tdico *dico, Pchar s, Pchar x, Bool err)
/* s= a subckt define line, with formal params.
   x= a matching subckt call line, with actual params 
*/
Begin
  short n,m,i,j,k,g,h, narg=0, ls, nest;
  Strbig(Llen,t);
  Strbig(Llen,u);
  Strbig(Llen,v);
  Strbig(Llen,idlist);
  Str(80,subname);
	  
  /***** first, analyze the subckt definition line */
  n=0; /* number of parameters if any */
  ls=length(s);
  j=spos("//",s);
  If j>0 Then pscopy(t,s,1,j-1) Else scopy(t,s) EndIf 
  stupcase(t); 
  j= spos("SUBCKT", t); 
  If j>0 Then
    j= j +6; /* fetch its name */
    While (j<ls) And (t[j]<=' ') Do Inc(j) Done
    While alfanum(t[j]) Do
      cadd(subname,t[j]); Inc(j) 
    Done 
  Else 
    err=message( dico," Not a subckt line!") 
  EndIf;
  i= spos("PARAMS:",t); 
  If i>0 Then 
    pscopy(t,t, i+7, length(t)); 
    While j=cpos('=',t), j>0 Do /* isolate idents to the left of =-signs */
      k= j-2; 
      While (k>=0) And (t[k]<=' ') Do Dec(k) Done
      h=k;
      While (h>=0) And alfanum(t[h]) Do Dec(h) Done
      If alfa(t[h+1]) And (k>h) Then /* we have some id */
        For m=(h+1); m<=k; Inc(m) Do 
          cadd(idlist,t[m])
        Done 
        sadd(idlist,"=$;");
        Inc(n);
      Else 
        message( dico,"identifier expected.")
      EndIf     
      pscopy(t,t, j+1, length(t));
    Done
  EndIf
  /***** next, analyze the circuit call line */
  If Not err Then
    narg=0;
    j=spos("//",x);
    If j>0 Then pscopy(t,x,1,j-1) Else scopy(t,x) EndIf 
    stupcase(t);
    ls=length(t);
    j= spos(subname,t); 
    If j>0 Then
      j=j + length(subname) -1; /* 1st position of arglist: j */
      While (j<ls) And ((t[j]<=' ') Or (t[j]==',')) Do Inc(j) Done 
      While j<ls Do /* try to fetch valid arguments */
        k= j; 
        scopy(u,"");
        If (t[k]==Intro) Then /* handle historical syntax... */
          If alfa(t[k+1]) Then 
            Inc(k)
          ElsIf t[k+1]=='(' Then /* transform to braces... */
            Inc(k); t[k]='{';
            g=k;  nest=1;
            While (nest>0) And (g<ls) Do
              Inc(g); 
              If t[g]=='(' Then Inc(nest) 
              ElsIf t[g]==')' Then Dec(nest)
              EndIf
            Done
            If (g<ls) And (nest==0) Then t[g]='}' EndIf
          EndIf
        EndIf
        If alfanum(t[k]) Then /* number, identifier */
          h=k; 
          While t[k] > ' ' Do Inc(k) Done
          pscopy(u,t, h+1, k-h); 
          j= k;    
        ElsIf t[k]=='{' Then
          getexpress(t,u, Addr(j)); 
          Dec(j); /* confusion: j was in Turbo Pascal convention */
        Else 
          Inc(j);
          If t[k]>' ' Then 
            scopy(v,"Subckt call, symbol "); 
            cadd(v,t[k]); 
            sadd(v," not understood");
            message( dico,v);
          EndIf 
        EndIf
        If NotZ(u[0]) Then 
          Inc(narg);
          k=cpos('$',idlist); 
          If k>0 Then /* replace dollar with expression string u */
            pscopy(v,idlist,1,k-1);
            sadd(v,u); 
            pscopy(u,idlist, k+1, length(idlist));
            scopy(idlist,v); 
            sadd(idlist,u); 
          EndIf
        EndIf
      Done
    Else 
      message( dico,"Cannot find called subcircuit") 
    EndIf 
  EndIf
  /***** finally, execute the multi-assignment line */
  dicostack(dico, Push);  /* create local symbol scope */
  If narg != n Then
    scopy(t," Mismatch: ");
    nadd(t,n); 
    sadd(t,"  formal but ");
    nadd(t,narg); 
    sadd(t," actual params."); 
    err= message( dico,t);
    message( dico,idlist);
  /* Else debugwarn(dico, idlist) */
  EndIf
  err= nupa_assignment(dico, idlist, 'N');      
  return err
EndFunc

Proc nupa_subcktexit( tdico *dico)
Begin
  dicostack(dico, Pop);
EndProc

