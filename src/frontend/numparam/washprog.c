/*       washprog.c                Copyright (C)  2002    Georg Post
 *
 *  This file is part of Numparam, see:  readme.txt  
 *  Free software under the terms of the GNU Lesser General Public License 
 */

/****  washprog:  trivial text substitution utility.   ****/

/* history: this was an exercise to make an 'intersection' language
   of C and Java, that would look like Basic. A complete failure, of course. 

   Now only used to clean my Basic/Pascal-contaminated C code.
   With the rules file below, it destroys all those macros of mine for
   quiche eaters, which seem offensive to C aficionados.

   Standard rules file needed : downgrad.txt 

   Typical command line: ./washprog -r downgrad washprog.c
  
 There is no printf. Console Output/Input primitives are as follows:
     wc ws wr wn wi wln rln
 The bare-bones string(=Pchar) manipulation library is this:
     pscopy streq str length upcase scopy sadd saddn cadd pos

Format of  substitution rules:

 s <string> <string>   substitute. use ""  around string if spaces inside.
 w <string> <string>   first string must be a whole word only
 m <macro1> <macro2>   macro substitution with args 1 2 3 ...
 u <macro1> <macro2>   macro with atomic args, no punctuation "(;,:)" inside.
 x <strng1> <strng2>   exclude text section from strng1 to strng2.
 a <mac1>   <mac2>     dynamically add a new macro rule, if table space left.

 string: may contain special chars:  ^A ... ^Z \n \"
 macro1: string with "placeholders" 1 2 ... 9, in this order
 macro2: may contain the "arguments" anywhere
         non-arg digits in macro2 are prefixed 0

 Heavy use of 3 string operations:
 -  pscopy() substring extraction.
 -  comparison:   match().
 -  spos() substring search

 added : special postprocessing for C to place the ; and } :
1.  any ';' following a ';' or '}' is wiped out.
2.  any ';' preceding a '}' is wiped out.
3.  any remaining ';'  on start of line is shifted to end of preceding one.
*/

#include <stdio.h>   /* NULL FILE fopen feof fgets fclose fputs fputc gets */
#include "general.h"

Cconst(nsub, 100+1)   /*max nbr of substitution rules */
Cconst(nargs, 11)     /*max number of macro args + 1*/
Cconst(wild,'æ')      /* wildcard character in patterns */
Cconst(joker,1)       /* one-character placeholder */
Cconst( Recursion, True) /* 20 % slower, re-substitute inside macro args */

Tarray(macargs, string, nargs) /* 0..9 macro copy args, 10: a wildcard */

  /* global vars */
short isr;     /* nb of substitution rules */
Bool cMode;    /* a scanning options: c language mode */
short lookmax; /* input lookahead max size */
Pfile fout;    /* file filled by: echoOut macroOut translate traduire */

Tarray(str40, char, 44)
Tarray(str80, char, 84)
Darray(search, str40,  nsub)
Darray(replace, str80, nsub)
Str(nsub, srule);
Str(nsub, wildcard);

/********* trivial io ***/

Proc wsf( Pchar s, short fmt)
Begin
  short k;
  For k=1; k<=fmt-length(s); Inc(k) Do 
    wc(' ') 
  Done
  ws(s)
EndProc

Proc wcf(char c, short fmt)
Begin
  short k;
  For k=1; k<=fmt-1; Inc(k) Do 
    wc(' ') 
  Done
  wc(c)
EndProc

Proc wif(long i, short fmt)
Begin /*default fmt=1*/
  Str(30, s);
  nadd(s,i); 
  wsf(s,fmt)
EndProc

Proc rln(Pchar s) /*  78 column limit */
Begin
  short i; Bool done; char c;
  short max=maxlen(s);
  If max>78 Then max=78 EndIf
  i=0; done=False; 
  scopy(s,"");
  While Not done Do
    c=fgetc(stdin);
    If (c>=' ') And (c<='~') And (i<max) Then
       cadd(s,c); Inc(i) 
    EndIf
    done= (c=='\n') Or (c==Cr)
  Done
EndProc

/*****************/

Proc saddn( Pchar s, Pchar t, short n)
Begin
  Strbig(Llen,u);
  short lt= length(t);
  If lt<= n Then 
    sadd(s,t) 
  Else
    pscopy(u,t,1,n); 
    sadd(s,u)
  EndIf
EndProc

Proc allocdata(void)
Begin /* prevent any string overflow */
  short i;
  For i=0; i<nsub; Inc(i) Do
    Sini(search[i]); 
    Sini(replace[i])
  Done 
EndProc

Proc setOptions(Pchar s)
/* command-line options c-mode and/or lookahead buffer size */
Begin
  short j,k; 
  Bool num; 
  short z; 
  char c;
/*-StartProc-*/
  ws("Options: ");
  For j=1; j<length(s); Inc(j) Do /*scan for option setting chars */
    If s[j]=='C' Then 
       cMode=True; ws("cMode ") 
    EndIf
    If s[j]=='L' Then /*redefine max lookahead length */
      z=0; 
      k= (short)(j+1);
      Repeat
        Inc(k); c=s[k]; 
        num= (c>='0') And (c<='9');
	If num Then z= (short)( 10*z+ c - '0') EndIf
      Until Not num EndRep
      If (z>lookmax) And (z<255) Then 
        lookmax= z 
      EndIf
      ws("Lookahead="); wi(lookmax);
    EndIf
  Done
  wln();
EndProc

/******** matching routines *******/

Proc copySpace(Pchar s, Pchar t, short a, short b) /* a,b>0 ! Pascal indexing */
Begin
/*echo any "nontrivial" whitespace t-->s */
  short lt,i,k, comment; 
  Bool leader; 
  char c;
/*-StartProc-*/ 
  scopy(s,"");  
  leader=False; /*leader space on new line...*/
  k=0; 
  comment=0; /* for C type whitespaces 1 And 2*/
  lt= length(t);
  If b>lt Then b=lt EndIf
  For i=(short)(a-1); i<b; Inc(i) Do
    c=t[i]; 
    If (c>0) And (c<' ') Then leader=True EndIf
    If cMode And (c=='/') And (t[i+1]=='*') Then comment=1 EndIf
    If ((c>0) And (c<' ')) Or (leader And (c==' ')) Or (comment>0) Then
      cadd(s,c); Inc(k); 
    EndIf
    If (comment==1) And (c=='/') And (t[i-1]=='*') Then comment=0 EndIf
  Done
EndProc

Func short skipCwhite(Pchar t, short j, short lt) /* assume C indexing */
Begin
/* skip any C And C++ type whitespace in t, from j to lt */
/* returns j-1 If current char is no white at all! */
  char c; 
  short comment; /*types 1 And 2! */
  /*t[j] may already be '/' ? */ comment=0;
  c=t[j]; /*If c>' ', we are done! */
  If (c>0) And (c<=' ') Then
    Repeat
      If (comment==0) And (c=='/') Then
        If t[j+1]=='*' Then 
          comment=1
        ElsIf t[j+1]=='/' Then 
          comment=2 
        EndIf
      ElsIf (comment==1) And (c=='/') And (t[j-1]=='*') Then 
        comment=0
      ElsIf (comment==2) And (c==Lf) Then 
        comment=0 
      EndIf
      Inc(j); c=t[j];
    Until (j>lt) Or ((comment==0) And (c>' ')) EndRep
  EndIf
  return (short)(j-1); /* return last white-matching char position */
EndProc

Func  Bool simple(Pchar s)
Begin /* check if no strange punctuations inside s */
  char c; 
  short i,ls; 
  Bool found;
/*-StartProc-*/
  ls=length(s); 
  i=0;
  Repeat c=s[i];
    found=(c=='(') Or (c==')') Or (c==',') Or (c==';') Or (c==':');
    Inc(i);
  Until found Or (i>=ls) EndRep
  return Not found;
EndFunc

Func Bool match(Pchar s, Pchar t, short n, short tstart)
Begin
/* test if t starts with substring s. 
   returns 0 If tstart is out of range. But n may be 0 ? 
   options:  Singlechar wildcards "?" 
*/
   short i,j,lt; 
   Bool ok;
/*-StartProc-*/
  i=0; j=tstart; 
  lt= length(t); 
  ok=(tstart<lt);
  While ok And (i<n) Do
    ok= (j<lt) And ((s[i]==t[j]) Or (s[i]==joker));
    Inc(i); Inc(j);
  Done
  return ok
EndFunc

Func short posi(Pchar sub, Pchar s)
Begin /*re-defines Turbo Pos, result Pascal compatible */
  short a,b,k; 
  Bool ok;
/*-StartProc-*/ 
  ok=False;
  a=length(sub); 
  b=(short)(length(s)-a); 
  k=0;
  If a>0 Then  /*Else return 0*/
    While (k<=b) And (Not ok) Do
      ok=match(sub,s, a,k); /*remark we must start at k=0 ! */
      Inc(k);
    Done
  EndIf
  If ok Then 
    return k 
  Else 
    return 0 
  EndIf
EndFunc

Func short matchwhite(Pchar s, Pchar t, short n, short tstart)
Begin
/* like match, but any whitespace in t matches space in s*/
  short i,j,lt; Bool ok;
/*-StartProc-*/ 
  i=0; j=tstart; 
  lt= length(t); 
  ok=(tstart<lt);
  While ok And (i<n) Do
    If s[i]==' ' Then /* always Ok, skip space in t */
      If cMode Then 
        j=skipCwhite(t,j,lt) 
      Else
        While (j<=lt) And (t[j]<=' ') And (t[j]>0) Do Inc(j) Done
        Dec(j);
      EndIf
      Repeat 
        Inc(j) 
      Until (j>=lt) Or (t[j]>' ') EndRep /*skip space in t*/
      Dec(j);
    Else
      ok= (j<=lt) And ((s[i]==t[j]) Or (s[i]==joker));
    EndIf
    Inc(i); Inc(j);
  Done
  If ok Then 
     return (short)(j-tstart) 
  Else 
    return (short)0 
  EndIf
EndFunc

Func short posizero(Pchar sub, Pchar s)
Begin /*another Pos */
/* substring search. like posi, but reject quotes & bracketed stuff */
  short a,b,k; 
  Bool ok; 
  short blevel; 
  char c;
/*-StartProc-*/ 
  ok=False;
  a=length(sub); 
  b=(short)(length(s)-a); 
  k=0; blevel=0;
  If a>0 Then /*Else return 0*/
    While (k<=b) And (Not ok) Do
      ok= (matchwhite(sub,s, a,k)>0);
      If (k<=b) And (Not ok) Then 
        c=s[k];
        If (c==')') Or (c==']') Or (c=='}') Then
          If c!=sub[0] Then Dec(blevel) EndIf /*negative level: fail!*/
	  If blevel<0 Then k=b EndIf
	ElsIf (c=='\'') Or (c=='\"') Then  /*skip quote */
          Repeat Inc(k) 
          Until (k>=b) Or (s[k]==c) EndRep
        ElsIf (c=='(') Or (c=='[') Or (c=='{') Then /*skip block*/
          Inc(blevel); /*counts the bracketing level */
          Repeat 
            Inc(k); c=s[k];
            If (c=='(') Or (c=='[') Or (c=='{') Then 
              Inc(blevel)
            ElsIf (c==')') Or (c==']') Or (c=='}') Then 
              Dec(blevel)
            EndIf
          Until (k>=b) Or (blevel==0) EndRep
        EndIf
      EndIf
      Inc(k);
    Done
  EndIf
  If ok Then 
    return k 
  Else 
    return 0 
  EndIf
EndFunc

Func short isMacro(Pchar s, char option, Pchar t, short tstart,
  string maccopy[] )
/* s= macro template, t=buffer, maccopy = arg Array 
   return value: number of characters matched,
   restrictive option: 'u'
   macro substitution args 1 2 3 ...9. 
   sample: bla1tra2gla3vla matches "bla ME tra YOU gla HIM vla" 
   substitute 1 by maccopy[1] etc 
*/
Begin
  Darray(ps, short, nargs+1)
  Word j,k,dk,ls, lst, lmt, jmax, pj; 
  Bool ok;
  char arg; 
  Strbig(Llen,u); 
  Str(40,st);
/* returns >0 If comparison Ok == length of compared Pchar */
/*-StartProc-*/  k=0;
  ok= (s[0]==t[tstart]); /* shortcut: how much does it accelerate ? some % */
  If ok Then
    ps[0]=0; 
    ps[nargs]=0; /*only 1..9 are valid data, 10 filler templates*/
    j=0;
    Repeat 
      Inc(j); arg= (char)(j+'0'); 
      ps[j]= cpos(arg,s);
    Until (j>=nargs) Or (ps[j]==0) EndRep
    ls= length(s); 
    ps[j]=(short)(ls+1); /*For last template chunk*/
    jmax=j; j=1; 
    k=0; lmt=0;
    Repeat
      pscopy(st,s, (Word)(ps[j-1]+1), (Word)(ps[j]-ps[j-1]-1) );
      /*j-th template Pchar*/  lst=length(st);
      If j==1 Then
        If option=='u' Then
          lmt= matchwhite(st,t,lst,tstart); 
          ok=(lmt>0) /*length of match in t*/
        Else 
          ok= match(st,t,lst,tstart) 
        EndIf
	If ok Then
          pscopy(u,t, (Word)(tstart+1), (Word)255); 
          pj=1 
        Else 
          pj=0 
        EndIf
      Else
        If option=='u' Then 
          pj= posizero(st,u);
          If pj>0 Then lmt= matchwhite(st,u, lst, (short)(pj-1)) EndIf
        Else 
          pj= posi(st,u) 
        EndIf  /* qs[j]= k+pj; is position in t*/
        ok=(pj>0);
      EndIf
      If ok Then
        If option=='u' Then
          If j==1 Then scopy(maccopy[0],"") EndIf
          saddn(maccopy[j-1],u, (Word)(pj-1));
          dk= (Word)(pj+lmt);
          copySpace(maccopy[j], t, 
            (Word)(tstart+k+pj), (Word)(tstart+k+dk));
          /* space in t[k+pj...k+dk] goes into maccopy[j] as a prefix. */
        Else
          pscopy(maccopy[j-1],u, (Word)1, (Word)(pj-1));
           /*the stuff preceding the marker*/
	  dk= (Word)(pj+lst);  /* start of unexplored part */
        EndIf
        pscopy(u,u, (Word)dk, (Word)length(u));  /*shift in the rest*/
        k= (Word)(k+dk-1);
      EndIf
      Inc(j)
    Until (j>jmax) Or (Not ok) EndRep
  EndIf
  If Not ok Then k=0 EndIf
  return k
EndFunc

Func short similar(Pchar s, char wilds, Pchar t,
  short tstart, string maccopy[] )
/* try to match s with t, then save the wildcard parts ins maccopy[] */ 
/* s=template, t=buffer, wilds= number of wildcards, maccopy=substitute */
/* return value: number of characters matched */
Begin
  Word j,k,ps,ls; 
  Bool ok;
  char endc; 
  Strbig(Llen,u);
/* returns >0  if comparison Ok = length of compared string */
/* char comparison, s may have wildcard regions with "æ" BUT 1 valid End */
/*-StartProc-*/ 
  ls=length(s);  
  k=0;
  If wilds==wild Then 
    ps= cpos(wild,s) 
  Else 
    ps=0 
  EndIf
  If ps==0 Then
    If match(s,t,ls,tstart) Then
      k=ls; 
      ps= cpos(joker,s); /*save joker's substitute*/
      If ps>0 Then 
        maccopy[nargs][0]=t[ps-1+tstart] 
      EndIf
    Else 
       k=0 
    EndIf
  Else   
    k= (Word)(ps-1);
    While s[k]==wild Do Inc(k) Done
    endc=s[k]; /*End char to detect, at length */
    ok= match(s,t, (short)(ps-1), tstart);
    If ok Then
      pscopy(u,t, (Word)(ps+tstart), (Word)255);
      j= cpos(endc, u); 
      ok=(j>0);
      If ok Then 
        k= (Word)(ps+j-1);
        pscopy(maccopy[nargs],t, (Word)(ps+tstart), (Word)(j-1));
      EndIf
    EndIf
    If Not ok Then k=0 EndIf
  EndIf
  return k
EndProc

Func short addSubList(Pchar s, short isr)  
/* add the rule s to the Rule list at isr */
Begin
  short j,ls; 
  char c,d,endc;
  Bool start,stop;
/*-StartProc-*/
  ls=length(s); /* must kill the Newline */
  endc=' ';
  While (ls>0) And (s[ls]<' ') Do Dec(ls) Done;
  s[ls+1]=' '; 
  s[ls+2]=0; /* add a space */
  If s[0]=='o' Then 
    setOptions(s)
  ElsIf (isr<nsub) And   (cpos(s[0],"swmuxa") >0) Then
    j=1;
    Inc(isr); 
    scopy(search[isr],""); scopy(replace[isr],"");
    srule[isr]=(s[0]);  
    wildcard[isr]=0;
      /*init search*/
    start=True; stop=False; 
    d=0;
    While Not stop Do 
      Inc(j); c=s[j];
      If start Then
        If c !=' ' Then
          start=False;
	  If c=='\"' Then endc=c Else endc=' ' EndIf
        EndIf
      Else 
        stop=(c==endc) 
      EndIf
      If Not (start Or (c==endc)) Then
        If c=='?' Then
          c=joker
        ElsIf (c=='^') And (s[j+1]>= ' ') Then
          Inc(j); c=s[j];
          If (c>='@') And (c<='_') Then 
             c= (char)(c-'@') 
          EndIf
        ElsIf (c=='\\') And (s[j+1]>= ' ') Then
          Inc(j); c=s[j];
          If c=='n' Then c= Cr; d=Lf EndIf
        EndIf
        cadd(search[isr],c);
        If (c==wild) Or (c==joker) Then 
          wildcard[isr]=c 
        EndIf
        If d!=0 Then
          cadd(search[isr],d); 
          d=0
        EndIf
      EndIf
    Done
    If endc!=' ' Then Inc(j) EndIf
          /*init replace*/
    start=True; stop=False;  
    d=0;
    While Not stop Do 
      Inc(j); c=s[j];
      If start Then
        If c!=' ' Then
          start=False;
          If c=='\"' Then endc=c Else endc=' ' EndIf
        EndIf
      Else 
        stop=(c==endc) 
      EndIf
      If Not (start Or (c==endc)) Then
        If c=='?' Then
          c=joker
        ElsIf (c=='^') And (s[j+1]>= ' ') Then
          Inc(j); c=s[j];
          If (c>='@') And (c<='Z') Then c= (char)(c-'@') EndIf
        ElsIf (c=='\\') And (s[j+1]>= ' ') Then
          Inc(j); c=s[j]; /*echo next char */
          If c=='n' Then c=Cr; d=Lf EndIf
        EndIf
        cadd(replace[isr],c);
        If d!=0 Then
          cadd(replace[isr],d); 
          d=0
        EndIf
      EndIf
    Done
    If endc !=' ' Then Inc(j) EndIf
  EndIf
  If isr>=nsub Then
    ws("No more room for rules."); wln() 
  EndIf
  return isr
EndFunc

Func Bool getSubList(Pchar slist)
/* read the search and substitution rule list */
Begin
  Strbig(Llen,s); 
  Pfile f;
  Bool done, ok;
/*-StartProc-*/
  cMode=False;
  lookmax= 80; /* or 250: handle 4 full lines  maximum ? */
  If Zero(slist[0])  Then 
    scopy(slist, "slist.txt") 
  EndIf
  f=fopen(slist,"rb"); 
  isr=0;
  done= (f == Null);
  ok= Not done;
  While Not done Do
    fgets(s,(short)80,f);
    isr=addSubList(s,isr);
    done= feof(f)
  Done
  If f != Null Then fclose(f) EndIf
  ws("Number of rules: "); 
  wi(isr); wln();
  return ok
EndFunc

Func Bool nonAlfa(char c)
Begin
  return ((c<'a') Or (c>'z')) And ((c<'A') Or (c>'Z'))
EndFunc

/********** optional output postprocessor **************/

/* the main translator calls these:
    washinit    to reset the postprocessor
    washchar    to output a char
    washstring  to output a string
    washflush   to terminate
*/

/*  C reformatter, keeping an eye on the following (modulo whitespace):
      ; }  Lf.  

 This is just a state machine, handling 3 rules using an output buffer obf. 
 <white> means space excluding \n,  and <white2>, space including newlines.
 Wanted: regular-expression scripts or tricks to do the same or better...
 
 Rule1:  <white>Lf<white>; --> ;<white>Lf<white>       states 2 3 
 Rule2:  ;<white2>;         --> ;<white2>              state  1  
 Rule3:  }<white2>;         --> }<white2>              state  1 
*/

Bool washmore= True;  /* flag that activates the postprocessor */
Strbig(Llen,obf);         /* output buffer */
short iobf=0;         /* its index */   
short wstate=0;       /* output state machine */

Proc washinit(void)
Begin
  iobf=0; 
  wstate=0
EndProc

Proc washchar(char c)
Begin  /* state machine receives one character */
  short i;
  If Not washmore Then /* never leave state 0 */
    fputc(c, fout)
  ElsIf wstate==0 Then /* buffer empty */
    If (c==';') Or (c=='}') Then
      iobf=0; obf[iobf]=c; 
      Inc(iobf); wstate=1
    ElsIf c<=' ' Then
      iobf=0; obf[iobf]=c; 
      Inc(iobf);
      If c==Lf Then wstate=3 Else wstate=2 EndIf
    Else  
      fputc(c, fout)
    EndIf
  ElsIf wstate==1 Then
    If c <= ' ' Then
      obf[iobf]=c; Inc(iobf)
    Else
      If c != ';' Then 
         obf[iobf]=c; Inc(iobf) 
      EndIf
      For i=0; i<iobf; Inc(i) Do 
        fputc(obf[i], fout) 
      Done
      iobf=0; 
      wstate=0
    EndIf
  ElsIf wstate==2 Then
    obf[iobf]=c; Inc(iobf); 
    If c==Lf Then  
      wstate=3
    ElsIf c<=' ' Then /* keep state */
    Else  
      For i=0; i<iobf; Inc(i) Do 
        fputc(obf[i], fout) 
      Done
      iobf=0; 
      wstate=0
    EndIf
  ElsIf wstate==3 Then
    obf[iobf]=c; Inc(iobf); 
    If c<=' ' Then /* keep state */
    Else
      If c==';' Then 
        Dec(iobf); fputc(c, fout) 
      EndIf
      For i=0; i<iobf; Inc(i) Do 
        fputc(obf[i], fout) 
      Done
      iobf=0; 
      wstate=0
    EndIf
  EndIf
EndProc

Proc washflush(void)
Begin
  short i;
  If NotZ(wstate) Then
    For i=0; i<iobf; Inc(i) Do 
      fputc(obf[i], fout) 
    Done
    iobf=0; 
    wstate=0
  EndIf
EndProc

Proc washstring( Pchar s)
Begin
  short i;
  For i=0; i<length(s); Inc(i) Do 
    washchar(s[i]) 
  Done
EndProc
   
/************* main part of translation filter  ***********/

Proc translate(Pchar bf);  /* recursion */

Proc echoOut(Pchar r, char isWild, string mac[] )
Begin
  short u; 
  Strbig(Llen,s);
/*-StartProc-*/
  If isWild !=0 Then
     u= cpos(isWild,r) 
  Else 
    u=0 
  EndIf
  If u==0 Then 
    washstring(r) 
  Else /*substitute with wildcard*/
    pscopy(s,r, (Word)1, (Word)(u-1)); washstring(s);
    If isWild==joker Then 
      washchar(mac[nargs][0])
    ElsIf Recursion Then 
      translate(mac[nargs])
    Else 
      washstring(mac[nargs]) 
    EndIf
    scopy(mac[nargs], "");
    pscopy(s,r, (Word)(u+1), (Word)40); 
    washstring(s);
  EndIf
EndProc

Proc macroOut(Pchar r, string mac[] )
Begin
/* substitutes "1"..."9", uses "0" as escape character*/
  char c; 
  short i,j; 
  Bool escape;
/*-StartProc-*/ 
  escape=False;
  For i=0; i<length(r); Inc(i) Do
    c=r[i]; 
    j= (short)(c-'0');
    If j==0 Then 
      escape=True /*And skip*/
    ElsIf ((j>0) And (j<nargs)) And (Not escape) Then
      If Recursion Then 
        translate(mac[j]) 
      Else 
        washstring(mac[j]) 
      EndIf
    Else 
      washchar(c); 
      escape=False 
    EndIf
  Done
EndProc

Proc makeNewRule(Pchar r, string mac[] )
Begin
/* substitutes "1"..."9", uses "0" as escape character*/
  char c; 
  short i,j; 
  Bool escape; 
  Strbig(Llen,s);
/*-StartProc-*/ 
  escape=False; 
  For i=0; i<length(r); Inc(i) Do
    c=r[i]; 
    j= (short)(c-'0');
    If j==0 Then 
      escape=True /*And skip*/
    ElsIf ((j>0) And (j<nargs)) And (Not escape) Then
      sadd(s,mac[j])
    Else 
      cadd(s,c); escape=False 
    EndIf
  Done
  isr= addSubList(s,isr)
EndProc

Proc translate(Pchar bff)
Begin /*light version, inside recursion only */
  Bool  done; 
  Strbig(Llen,bf);
  Darray(mac, string, nargs)
  Bool ok; 
  short i,sm; 
  char lastBf1; 
  Word nbrep;
/*-StartProc-*/
  For i=0; i<nargs; Inc(i) Do 
    Sini(mac[i]) 
  Done
  nbrep=0;
  done= Zero(bff[0]); 
  lastBf1=' ';
  If Not done Then scopy(bf,bff) EndIf
  While Not done Do
    i=1; 
    ok=False; sm=0;
    While (i<=isr) And (Not ok) Do /*search For 1st match*/
      If (srule[i]=='m') Or (srule[i]=='u') Then
        If alfa(lastBf1) And (alfa(search[i][0])) Then 
          sm=0 /*inside word*/
        Else 
          sm= isMacro(search[i], srule[i], bf, (short)0,mac) 
        EndIf
      Else 
        sm=similar(search[i],wildcard[i],bf, (short)0, mac) 
      EndIf
      ok=sm>0;
      If ok And (srule[i]=='w') Then
	ok=nonAlfa(lastBf1) And nonAlfa(bf[sm])
      EndIf
      If Not ok Then Inc(i) EndIf
    Done
    If ok Then
      If (srule[i]=='m') Or (srule[i]=='u') Then 
        macroOut(replace[i], mac)
      Else 
        echoOut(replace[i],wildcard[i], mac) 
      EndIf
      lastBf1=bf[sm-1]; pscopy(bf,bf, (Word)(sm+1), (Word)255);
      Inc(nbrep);
    Else
      lastBf1=bf[0]; 
      washchar(lastBf1);
      pscopy(bf,bf, (Word)2, (Word)255);
    EndIf
    done= Zero(bf[0])
  Done
EndProc

Proc translator( Pchar fname)
/* checks list of possible substitution rules sequentially.
   Does the first that matches. Option: recursion.
   BUG: is very slow.
*/
Begin
  Strbig(Llen, outname); Strbig(Llen,bf);
  Bool done;
  Darray( mac, string, nargs)
  Pfile fin;
  Bool ok; 
  short i,sm, exclusion, idot; 
  char c,lastBf1; 
  Word nbrep,nline;
/*-StartProc-*/
  For i=0; i<nargs; Inc(i) Do 
    Sini(mac[i]) 
  Done
  nbrep=0; 
  nline=0;
  exclusion=0; /* will be >0 if an exclusion rule is active */
  fin=fopen( fname, "rb");
  scopy(outname, fname);
  idot= cpos('.',outname);
  If idot <= 8 Then /* room for underbar prefix, even in Ms-dos */
    cins(outname,'_')
  ElsIf NotZ(outname[0]) Then /* just erase first char */
    outname[0] = '_'
  Else
    scopy(outname,"washprog.out")
  EndIf
  fout=fopen( outname,"wb");
  washinit();
  done= (fin == Null) Or (fout == Null);  
  scopy(bf,"");  
  lastBf1=' ';
  /* lookmax=80; handle a line maximum ! */
  While Not done Do
    c=' ';
    While (c !=0) And (length(bf)<lookmax) Do /*refill buffer*/
      If Not feof(fin) Then
	c=fgetc(fin);
	If (c== Cr) Or (c== Lf) Then 
          Inc(nline);
	  If odd(nline) Then wc('.') EndIf
          If (nline Mod 150)==0 Then wln() EndIf
        EndIf
        If (c==0) Or feof(fin) Then c=' ' EndIf /*== space*/
      Else 
        c=0 
      EndIf
      If NotZ(c) Then cadd(bf,c) EndIf
    Done
    ok=False; 
    sm=0; i=0;
    If exclusion>0 Then 
      i=exclusion;
      sm=similar(replace[i], (char)0, bf, (short)0, mac);
      ok= sm>0
    EndIf
    If Zero(exclusion) Then
      i=1;
      While (i<=isr) And (Not ok) Do /*search for 1st match*/
        If (srule[i]=='m') Or (srule[i]=='u') Or (srule[i]=='a') Then
          If alfa(lastBf1) And (alfa(search[i][0])) Then 
            sm=0 /*inside word*/
          Else 
            sm= isMacro(search[i], srule[i], bf, (short)0,mac) 
          EndIf
        Else 
          sm=similar(search[i],wildcard[i],bf, (short)0, mac) 
        EndIf
        ok=sm>0;
        If ok And (srule[i]=='w') Then
          ok=nonAlfa(lastBf1) And nonAlfa(bf[sm])
        EndIf
        If Not ok Then Inc(i) EndIf
      Done
    EndIf
    If ok Then
      If (srule[i]=='m') Or (srule[i]=='u') Then
        macroOut(replace[i], mac)
      ElsIf srule[i]=='x' Then
	If Zero(exclusion) Then 
          exclusion=i 
        Else 
          exclusion=0 
        EndIf
      ElsIf srule[i]=='a' Then
        makeNewRule(replace[i],mac)
      Else
        echoOut(replace[i],wildcard[i],mac) 
      EndIf
      lastBf1=bf[sm-1]; pscopy(bf,bf, (Word)(sm+1), (Word)lookmax);
      Inc(nbrep);
    Else
      lastBf1=bf[0];
      If Zero(exclusion) Then washchar(lastBf1) EndIf;
      pscopy(bf,bf, (Word)2, (Word)lookmax);
       /*avoid this time-consuming buffer shuffling ?*/
    EndIf
    done= Zero(bf[0]);
  Done
  If fout !=Null Then 
    washflush(); 
    fputc('\n', fout); 
    fclose(fout) 
  EndIf
  If fin !=Null Then fclose(fin) EndIf
  ws("Lines: "); wi(nline); 
  ws("  Replacements: "); 
  wi(nbrep); wln();
EndProc

Func int main( int argc, Pchar argv[])
Begin
  Str(80,dico);
  short istart= 1;
  Bool ok= True;
/*-StartProc-*/
  allocdata();
  scopy(dico,"downgrad"); /* default rules file */
  ws(" washprog: A text substitution utility"); wln();
  If (argc>2) And steq(argv[1],"-r") Then 
    scopy(dico,argv[2]); 
    istart= 3; 
/*
  Else
    ws("Dictionary file (.TXT automatic): "); 
    rln(dico);
*/
  EndIf
  If spos(".txt",dico) <=0 Then 
    sadd(dico,".txt") 
  EndIf
  ok= getSubList(dico); /*list of substitution rules */
  While ok And (istart< argc) Do
    If argv[istart][0] != '_' Then /* leading underbar not accepted */
      translator( argv[istart])
    EndIf
    Inc(istart)
  Done
  return 0
EndFunc

