/*       nupatest.c                Copyright (C)  2002    Georg Post
 *
 *  This file is part of Numparam, see:  readme.txt  
 *  Free software under the terms of the GNU Lesser General Public License 
 */

/****   test executable for the numparam library  ****/
/****   usage:  nupatest <filename.cir>           ****/

#include <stdio.h>

#include "general.h"
#include "numparam.h"

Cconst(pfxsep,'_')  /* official prefix separator is ':' not '_'  ! */

Darray(buff, Pchar, Maxline) /* input lines */
Darray(buf2, Pchar, Maxline) /* stripped lines */
Darray(pxbuf, Pchar, Maxline) /* prefix for subnodes */
Darray(runbuf, short, Maxckt) /* index list of expanded circuit */
Darray(pindex, short, Maxckt) /* prefix index list  */
short irunbuf= 0; /* count lines of runbuf */
short ipx=0;      /* count prefixes in pxbuf */  

/*  
   this toy imitates the Spice subcircuit expansion.
 To ckeck against Genuine Spice, use the 'listing expand' JDML command
 Bug1: model or device call with parameters: incorrectly parsed
    needs a database about _optional_ model/device pars...
    better: Enter all .model identifiers in the symbol table !
 Bug2: nested subckt declarations, local .models: might be dangerous.
   expanded circuit lines, device/node names: 
 any line that starts with a letter (device): splice the prefix in
 any node that isnt a formal param: add the prefix
 any formal param node: substitute actual params and their prefixes

Node and subdevice references for prefixing:

deviceletter[n] is a device type prefix
nbofnodes   [n] is the number of "node-type" arguments that follow.
nbsubdevice [n] is the number of "subdevices" for splice-in prefix.

To solve the Q ambiguity, forbid any model identifiers as node names.

Bug3:
In arbitrary dependent sources, we must parse v(,) and i(,) expressions
and substitute node/device name arguments.

*/

Func short runscript( tdico *dico, Pchar prefix,
   short istart, short istop, short maxnest)
/* recursive top-down expansion: circuit --> list of line numbers */
/* keep it simple,stupid  compared to Spice's code */
/* prefix: inherited string for node & device prefixing */
/* istart, istop: allowed interval in table buf[], buf2[]. */
/* return value: number of lines included */
Begin
  short i,j, idef, nnest, nline, dn, myipx;
  Strbig(Llen, subpfx); /* subckt prefix */ 
  Str(80, subname);
  char c;
  Bool done= False;
  i=istart; 
  nline=0;
  Inc(ipx); myipx= ipx; /* local copy */
  pxbuf[ipx]= newstring( length(prefix));
  scopy( pxbuf[ipx], prefix);
  While (maxnest>0) And (i<istop) And (Not done) Do
    c= dico->category[i];
    If c=='U' Then 
      done=True; /* subcircuit end. Keep as a comment? */
      buf2[i][0]='#';
    EndIf
    If c=='S' Then        /* skip nested subcircuits */
      nnest=1;
      Repeat 
        Inc(i); c= dico->category[i];
        If c=='S' Then 
          Inc(nnest) 
        ElsIf c=='U' Then 
          Dec(nnest) 
        EndIf
      Until (nnest<=0) Or (i>=istop) EndRep
    ElsIf c=='X' Then                   /* recursion here ! */
      runbuf[irunbuf]= i;
      pindex[irunbuf]= myipx;
      Inc(irunbuf); Inc(nline);
      /* keep out-commented X line for parameter passing */
      idef = findsubckt( dico, buf2[i], subname); 
      buf2[i][0]= '*'; 
      If idef>0 Then
        scopy(subpfx, prefix); 
        cadd(subpfx, pfxsep);
        j=1; /* add the instance name from buf2[i] */
        While buf2[i][j] > ' ' Do
          cadd( subpfx, buf2[i][j]); Inc(j)
        Done 
        dn= runscript(dico, subpfx, idef+1, istop, maxnest-1);
        nline= nline+dn; 
      Else /* FIXME: error message here! */
        ws("cannot find subckt "); ws(buf2[i]); wln(); 
      EndIf
    ElsIf (c != '?') And NotZ(buf2[i][0]) Then         
      /*  keep any other valid non-empty line, and its prefix pointer */
      runbuf[irunbuf]= i;
      pindex[irunbuf]= myipx;
      Inc(irunbuf); Inc(nline);
    EndIf 
    Inc(i);
  Done  
  return nline
EndProc

Proc gluepluslines( short imax)
/* general sweep to eliminate continuation lines */
Begin
  short i,j,k, ls, p;
  Strbig(Llen,s);
  i=1;
  While i<= imax Do
    If (buff[i][0]=='+') And (i>1) Then 
      j= i-1; 
      While (i < imax) And (buff[i+1][0]=='+') Do Inc(i) Done
      /* the lines j+1 ... i are continuation lines to j */
      For k=j; k<=i; Inc(k) Do 
        ls=length(s);
        sadd(s, buff[k]);
        p= spos("//",s);
        If p>0 Then pscopy(s,s, 1,p-1) EndIf
        If ls>0 Then s[ls]=' ' EndIf /* erase the + */
      Done
      ls= length(s);
      If ls> 80 Then
        Dispose(buff[j]);
        buff[j]=newstring(ls)
      EndIf
      scopy(buff[j], s)
    EndIf  
    Inc(i)
  Done
EndProc

#if 0	// sjb - this is in mystring.c
Proc rs(Pchar s) /*  78 coumn limit */
Begin
  short i; 
  Bool done; 
  char c;
  short max=maxlen(s);
  If max>78 Then max=78 EndIf
  i=0; done=False; 
  scopy(s,"");
  While Not done Do
    c=fgetc(stdin);
    If (c>=' ')And(c<='~') And (i<max) Then 
      cadd(s,c); Inc(i) 
    EndIf
    done= (c==Lf) Or (c==Cr)
  Done
EndProc
#endif

Proc fwrites(Pfile f, Pchar s)
Begin 
  fputs(s,f) 
EndProc

Proc fwriteln(Pfile f)
Begin 
  fputc('\n',f) 
EndProc

Intern
Proc freadln(Pfile f, Pchar s, short max)
Begin 
  short ls;
  freadstr(f,s,max); 
  ls=length(s);
  If feof(f) And (ls>0) Then 
    pscopy(s,s,1,ls-1) 
  EndIf /* kill EOF character */
EndProc

Proc wordinsert(Pchar s, Pchar w, short i)
/* insert w before s[i] */
Begin
  Strbig(Llen,t);
  short ls=length(s);
  pscopy(t,s,i+1,ls); pscopy(s,s,1,i);
  sadd(s,w); sadd(s,t);
EndProc

Func short worddelete(Pchar s, short i)
/* delete word starting at s[i] */
Begin
  Strbig(Llen,t);
  short ls= length(s);
  short j=i;
  While (j<ls) And (s[j]>' ') Do Inc(j) Done
  pscopy(t,s,j+1,ls);
  pscopy(s,s,1,i); 
  sadd(s,t);
  return j-i /* nb of chars deleted */
EndProc

Func short getnextword(Pchar s, Pchar u, short j)
Begin
  short ls,k;
  ls= length(s);
  k=j;
  While (j<ls) And (s[j] >  ' ') Do Inc(j) Done /* skip current word */ 
  pscopy(u, s,  k+1, j-k);
  While (j<ls) And (s[j] <= ' ') Do Inc(j) Done
  return j
EndFunc

Func short inwordlist(Pchar u, Pchar wl)
/* suppose wl is single-space separated, plus 1 space at start and end. */ 
Begin
  short n,p,k;
  Str(80,t);
  n=0;
  ccopy(t,' '); sadd(t,u); cadd(t,' ');
  p= spos(t,wl);
  If p>0 Then
    For k=0; k<p; Inc(k) Do
      If wl[k] <= ' ' Then Inc(n) EndIf
    Done
  EndIf
  return n
EndFunc

Proc takewordlist(Pchar u, short k, Pchar wl)
Begin
  short i,j,lwl;
  lwl= length(wl);
  i=0; j=0;
  scopy(u,"");
  While (i<lwl) And (j<k ) Do
    If wl[i] <= ' ' Then Inc(j) EndIf
    Inc(i)
  Done
  If j==k Then /* word has been found and starts at i */
    While wl[i]>' ' Do
      cadd(u,wl[i]); Inc(i)
    Done
  EndIf
EndProc

Pchar deviceletter= "RLCVIBSGETOUWFHDQKJZM";
Pchar nbofnodes   = "222222444443222240334";
Pchar nbsubdevice = "000000000000111002000";

Proc prefixing(Pchar s, Pchar p, Pchar formals, Pchar actuals,
   char categ, tdico *dic)
/* s is a line in expanded subcircuit. 
   p is the prefix to be glued anywhere .
   assume that everything except first and last word in s may be a node. 
   formals: node parameter list of a subckt definition line 
   actuals: substitutes from the last X... call line (commented-out) 
   subdevices (L belonging to a K line, for example) must be within the
   same subckt, they get the same prefix splice-in.
   There is a kludge for Q lines (may have 3 or 4 nodes, you never know).
Reminder on Numparam symbols:
     naming convention: subckt,model,numparam and node names must be unique.
     cannot re-use a model name as a param name elsewhere, for example.
*/
Begin
  short i,j,k,ls, jnext, dsize;
  short dtype, nodes, subdv;
  Bool done;
  char leadchar;
  Str(80,u); Str(80,v); Str(80,pfx);
  i=0; ls=length(s);
  While (i<ls) And (s[i]<=' ') Do Inc(i) Done
  If alfa(s[i]) Or (categ=='X') Then /* splice in the prefix and nodelist */
    wordinsert(s,p, i+1);
    j= getnextword(s,u,i); 
    done=False;
    If p[0]== pfxsep Then
      pscopy(pfx,p, 2, length(p)) 
    Else 
      scopy(pfx,p)
    EndIf
    leadchar=upcase(s[i]); 
    dtype= cpos( leadchar, deviceletter) -1 ;
    If dtype >= 0 Then 
      nodes= nbofnodes[dtype] - '0';
      subdv= nbsubdevice[dtype] - '0';
    Else
      nodes=999; subdv=0;
    EndIf
    While Not done Do
      jnext= getnextword(s,u,j);
      done=(jnext >= length(s)); /* was the last one, do not transform */
       /* bug: are there semilocal nodes ? in nested subckt declarations ? */
      If (leadchar=='Q') And (Not done) Then /* BJT: watch non-node name */
        scopy(v,u); stupcase(v);
        done=  getidtype(dic, v) == 'O'; /* a model name stops the node list */
      EndIf 
      If (Not done) And (nodes>0) Then /* transform a node name */
        k= inwordlist(u, formals);
        If (k>0) Then   /* parameter node */
          dsize= - worddelete(s,j);
          takewordlist(u,k, actuals);
          wordinsert(s,u,j); 
          dsize= dsize + length(u);
        ElsIf stne(u,"0") Then  /* local node */
          wordinsert(s,pfx,j);
          dsize= length(pfx);
        Else dsize=0 EndIf
      ElsIf (Not done) And (subdv >0) Then /* splice a subdevice name */
        wordinsert(s,p,j+1);
        dsize= length(p);
      EndIf
      j= jnext + dsize; /*  jnext did shift ...*/
      If nodes >0 Then Dec(nodes)
      ElsIf subdv >0 Then Dec(subdv)
      EndIf
      done= done Or (Zero(nodes) And Zero(subdv));
    Done
  EndIf
EndProc

Proc getnodelist(Pchar form, Pchar act, Pchar s, tdico *dic, short k)
/* the line s contains the actual node parameters, between 1st & last word */
Begin
  short j,ls, idef;
  Str(80,u); Strbig(Llen,t);
  ccopy(act,' '); ccopy(form,' ');
  j=0; ls= length(s);
  j= getnextword(s,u,j);
  While j<ls Do
    j= getnextword(s,u,j); 
    If j<ls Then sadd(act,u); cadd(act,' ') EndIf
  Done
  /* now u already holds the subckt name if all is ok ? */
  idef = findsubckt( dic, buf2[k], u); 
  /* line buf2[idef] contains: .subckt name < formal list > */ 
  If idef>0 Then 
    scopy(t, buf2[idef]) 
  Else
    ws("Subckt call error: "); ws(s); wln();
  EndIf
  j=0; ls= length(t);
  j= getnextword(t,u,j); 
  j= getnextword(t,u,j); 
  While j<ls Do
    j= getnextword(t,u,j); 
    sadd(form,u); cadd(form,' ');
  Done
EndProc

Proc nupa_test(Pchar fname, char mode)
/* debugging circuit expansion run. mode='w': write ouput file */
/* bugs in nupa_eval(), and for nested subckt definitions !?! */
Begin
  Pfile tf, fout;
  tdico * dic; /* dictionary data pointer */
  Strbig(Llen,s);
  Str(80, prefix);
  /* Strbig(Llen, formals); Strbig(Llen,actuals); */
  Darray(formals, Pchar, 10)
  Darray(actuals, Pchar, 10)
  short i, j, k, nline, parstack;
  For i=0; i<Maxline; Inc(i) Do /* allocate string storage */
    buff[i]= newstring(80);
    buf2[i]= Null;
    pxbuf[i]= Null 
  Done
  For i=0; i<10; Inc(i) Do
    formals[i]= newstring(250);
    actuals[i]= newstring(250);
  Done
  i=0; parstack=0;
  tf=fopen( fname, "r");
  If tf != Null Then
    While (Not feof(tf)) And ((i+1) < Maxline) Do
      Inc(i);
      freadln(tf, buff[i], 80); /* original data */
    Done
    fclose(tf);
  Else
    ws("Cannot find "); ws(fname); wln();
  EndIf
  /* continuation lines are glued at this stage, so they can be ignored
     in all the subsequent manipulations.
  */
  gluepluslines(i); /* must re-allocate certain buff[i]  */ 
  nupa_signal(NUPADECKCOPY, fname);
  dic= nupa_fetchinstance(); /* bug: should have a task handle as arg */
  For j=1; j<=i; Inc(j) Do
    buf2[j]= nupa_copy(buff[j], j); /* transformed data */
  Done
  nupa_signal(NUPASUBDONE, Null);
  nline= runscript(dic, "", 1,i, 20); /* our own subckt expansion */
  /* putlogfile(' ',nline," expanded lines"); */
  If mode=='w' Then
    i= cpos('.', fname); 
    pscopy(s, fname, 1, i);
    sadd(s,"out");
    fout= fopen(s, "w");
  Else
    fout= Null
  EndIf
  For j=0; j<irunbuf; Inc(j) Do
    k= runbuf[j]; 
    If buf2[k] != Null Then 
      scopy(s, buf2[k]);
      nupa_eval(s, k);
      scopy(prefix,pxbuf[pindex[j]]); 
      If NotZ(prefix[0]) Then cadd(prefix, pfxsep) EndIf 
      prefixing(s, prefix, formals[parstack], actuals[parstack],
           dic->category[k], dic);  
      If dic->category[k] == 'X' Then 
        If parstack< (10-1) Then Inc(parstack) EndIf
        getnodelist(formals[parstack], actuals[parstack], s, dic,k);
        /*dbg: ws("Form: "); ws(formals[parstack] ); wln(); */
        /*dbg: ws("Actu: "); ws(actuals[parstack]); wln(); */
      ElsIf dic->category[k]=='U' Then /* return from subckt */
        If parstack>0 Then Dec(parstack) EndIf
      EndIf
      If fout != Null Then
        fwrites(fout, s); fwriteln(fout)
      EndIf
    EndIf
  Done 
  If fout != Null Then fclose(fout) EndIf
  nupa_signal(NUPAEVALDONE, Null); /* frees the buff[i] */
  For i= 10-1; i>=0; Dec(i) Do
    Dispose(actuals[i]);
    Dispose(formals[i]);
  Done
  For i= Maxline -1; i>=0; Dec(i) Do
    Dispose(pxbuf[i]);  
    Dispose(buf2[i]); 
    /* Dispose(buff[i]) done elsewhere */
  Done
EndProc

Func int main(int argc, Pchar argv[])
Begin
  Str(80,fname);
  If argc>1 Then 
    scopy(fname, argv[1])
  Else 
    scopy(fname,"testfile.nup") 
  EndIf
  nupa_test(fname, 'w');
  return 0
EndFunc

