/*       spicenum.c                Copyright (C)  2002    Georg Post
 *
 *  This file is part of Numparam, see:  readme.txt  
 *  Free software under the terms of the GNU Lesser General Public License 
 */

/* number parameter add-on for Spice.
   to link with mystring.o, xpressn.o (math formula interpreter), 
   and with Spice frontend src/lib/fte.a . 
   Interface function nupa_signal to tell us about automaton states.
Buglist (some are 'features'):
  blank lines get category '*' 
  inserts conditional blanks before or after  braces 
  between .control and .endc, flags all lines as 'category C', dont touch. 
  there are reserved magic numbers (1e9 + n) as placeholders
  control lines must not contain {} . 
  ignores the '.option numparam' line planned to trigger the actions
  operation of .include certainly doesnt work 
  there are frozen maxima for source and expanded circuit size.
Todo:
  add support for nested .if .elsif .else .endif controls.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#ifdef __TURBOC__
#include <process.h>   /* exit() */
#endif

#include "general.h"
#include "numparam.h"
#include "ngspice.h"

extern void txfree(void *ptr);

/* Uncomment this line to allow debug tracing */
/*#define TRACE_NUMPARAMS*/

/*  the nupa_signal arguments sent from Spice:

   sig=1: Start of the subckt expansion.
   sig=2: Stop of the subckt expansion.
   sig=3: Stop of the evaluation phase.
   sig=0: Start of a deck copy operation

  After sig=1 until sig=2, nupa_copy does no transformations.
  At sig=2, we prepare for nupa_eval loop.
  After sig=3, we assume the initial state (clean).

  In Clean state, a lot of deckcopy operations come in and we
  overwrite any line pointers, or we start a new set after each sig=0 ?
  Anyway, we neutralize all & and .param lines  (category[] array!)
  and we substitute all {} &() and &id placeholders by dummy numbers. 
  The placeholders are long integers 1000000000+n (10 digits, n small).

*/
/**********  string handling ***********/

#define PlaceHold 1000000000L
Intern long placeholder= 0;

#ifdef NOT_REQUIRED /* SJB - not required as front-end now does stripping */
Intern
Func int stripcomment( Pchar s)
/* allow end-of-line comments in Spice, like C++ */
Begin
  int i,ls;
  char c,d;
  Bool stop;
  ls=length(s); 
  c=' '; i=0; stop=False;
  While (i<ls) And Not stop Do 
    d=c; 
    Inc(i); c=s[i-1];
    stop=(c==d) And ((c=='/')Or(c=='-'));  /* comments after // or -- */
  Done
  If stop Then 
    i=i-2; /*last valid character before Comment */
    While (i>0)And (s[i-1]<=' ') Do Dec(i) Done; /*strip blank space*/
    If i<=0 Then 
      scopy(s,"") 
    Else 
      pscopy(s,s,1,i) 
    EndIf
  Else 
    i= -1 
  EndIf
  return i /* i>=0  if comment stripped at that position */
EndFunc
#endif /* NOT_REQUIRED */

Intern
Proc stripsomespace(Pchar s, Bool incontrol)
Begin
/* iff s starts with one of some markers, strip leading space */
  Str(12,markers);
  int i,ls;
  scopy(markers,"*.&+#$"); 
  If Not incontrol Then 
    sadd(markers,"xX") 
  EndIf
  ls=length(s); i=0;
  While (i<ls) And (s[i]<=' ') Do Inc(i) Done
  If (i>0) And (i<ls) And (cpos(s[i],markers) >0) Then
    pscopy(s,s,i+1,ls)
  EndIf
EndProc

#if 0  /* unused? */
Proc partition(Pchar t)
/* t is a list val=expr val=expr .... Insert Lf-& before any val= */
/* the Basic preprocessor doesnt understand multiple cmd/line */
/* bug:  strip trailing spaces */
Begin
  Strbig(Llen,u);
  int i,lt,state;
  char c;
  cadd(u,Intro); 
  state=0; /* a trivial 3-state machine */
  lt=length(t);  
  While t[lt-1] <= ' ' Do Dec(lt) Done
  For i=0; i<lt; Inc(i) Do 
    c=t[i];
    If c=='=' Then 
      state=1
    ElsIf (state==1) And (c==' ') Then 
      state=2
    EndIf
    If state==2 Then 
      cadd(u,Lf); cadd(u,Intro); 
      state=0 
    EndIf
    cadd(u,c)
  Done
  scopy(t,u);
  For i=0; i<length(t); Inc(i) Do /* kill braces inside */
    If (t[i]=='{') Or (t[i]=='}') Then 
       t[i]=' '   
    EndIf
  Done
EndProc
#endif

Intern
Func int stripbraces( Pchar s)
/* puts the funny placeholders. returns the number of {...} substitutions */
Begin
  int n,i,nest,ls,j;
  Strbig(Llen,t);
  n=0; ls=length(s); 
  i=0;
  While i<ls Do
    If s[i]=='{' Then /* something to strip */
      j= i+1; nest=1; 
      Inc(n);
      While (nest>0) And (j<ls) Do
        If s[j]=='{' Then 
          Inc(nest) 
        ElsIf s[j]=='}' Then 
          Dec(nest) 
        EndIf
        Inc(j)
      Done
      pscopy(t,s,1,i);
      Inc(placeholder); 
      If t[i-1]>' '  Then cadd(t,' ') EndIf
      nadd(t, PlaceHold + placeholder); 
      cadd(t,' ');  // add extra character to increase number significant digits for evaluated numbers
      cadd(t,' ');
      cadd(t,' ');
      cadd(t,' ');
      cadd(t,' ');
      cadd(t,' ');
      cadd(t,' ');
      If s[j]>=' ' Then cadd(t,' ') EndIf
      i=length(t);
      pscopy(s,s, j+1, ls); 
      sadd(t,s); 
      scopy(s,t); 
    Else 
      Inc(i) 
    EndIf 
    ls=length(s);
  Done
  return n
EndFunc

Intern 
Func int findsubname(tdico * dico,  Pchar s)
/* truncate the parameterized subckt call to regular old Spice */
/* scan a string from the end, skipping non-idents and {expressions} */
/* then truncate s after the last subckt(?) identifier */
Begin
  Str(80, name);
  int h,j,k,nest,ls;
  Bool found;
  h=0; 
  ls=length(s);
  k=ls; found=False;		    
  While (k>=0) And (Not found)  Do /* skip space, then non-space */
    While (k>=0) And (s[k]<=' ') Do Dec(k) Done; 
    h=k+1; /* at h: space */
    While (k>=0) And (s[k]>' ')  Do 
      If s[k]=='}' Then 
        nest=1; 
        Dec(k);
        While (nest>0) And (k>=0) Do
          If s[k]=='{' Then 
            Dec(nest) 
          ElsIf s[k]=='}' Then 
            Inc(nest) 
          EndIf
          Dec(k)
        Done
        h=k+1; /* h points to '{' */
      Else 
        Dec(k) 
      EndIf;
    Done
    found = (k>=0) And alfanum(s[k+1]); /* suppose an identifier */
    If found Then /* check for known subckt name */
      scopy(name,""); j= k+1;
      While alfanum(s[j]) Do 
        cadd(name, upcase(s[j])); Inc(j) 
      Done
      found=  (getidtype(dico, name) == 'U');
    EndIf 
  Done		    
  If found And (h<ls) Then 
    pscopy(s,s,1,h) 
  EndIf
  return h;
EndFunc

Intern
Proc modernizeex( Pchar s)
/* old style expressions &(..) and &id --> new style with braces. */
Begin
  Strbig(Llen,t);
  int i,state, ls;
  char c,d;
  i=0; state=0;
  ls= length(s);
  While i<ls Do
    c= s[i]; d=s[i+1];
    If Zero(state) And (c==Intro) And (i>0) Then
      If d=='(' Then 
        state=1; Inc(i); c='{'
      ElsIf  alfa(d) Then
        cadd(t,'{'); Inc(i);
        While alfanum(s[i]) Do 
          cadd(t,s[i]); Inc(i)
        Done
        c='}'; Dec(i);
      EndIf
    ElsIf NotZ(state) Then
      If c=='(' Then
        Inc(state)
      ElsIf c==')' Then
        Dec(state)
      EndIf
      If Zero(state) Then /* replace ) by terminator */
        c='}';
      EndIf
    EndIf
    cadd(t,c);
    Inc(i)
  Done
  scopy(s,t);
EndProc

Intern
Func char transform(tdico * dico, Pchar s, Bool nostripping, Pchar u)
/*         line s is categorized and crippled down to basic Spice
 *         returns in u control word following dot, if any 
 * 
 * any + line is copied as-is.
 * any & or .param line is commented-out.
 * any .subckt line has params section stripped off
 * any X line loses its arguments after sub-circuit name
 * any &id or &() or {} inside line gets a 10-digit substitute.
 *
 * strip  the new syntax off the codeline s, and
 * return the line category as follows:
 *   '*'  comment line
 *   '+'  continuation line
 *   ' '  other untouched netlist or command line
 *   'P'  parameter line, commented-out; (name,linenr)-> symbol table.
 *   'S'  subckt entry line, stripped;   (name,linenr)-> symbol table.
 *   'U'  subckt exit line
 *   'X'  subckt call line, stripped
 *   'C'  control entry line
 *   'E'  control exit line
 *   '.'  any other dot line
 *   'B'  netlist (or .model ?) line that had Braces killed 
 */
Begin
  Strbig(Llen,t);  
  char category; 
  int i,k, a,n;

  stripsomespace(s, nostripping);
  modernizeex(s);    /* required for stripbraces count */
  scopy(u,"");
  If s[0]=='.' Then /* check Pspice parameter format */
    scopy_up(t,s); 
    k=1;
    While t[k]>' ' Do 
      cadd(u, t[k]); Inc(k) 
    Done
    If ci_prefix(".PARAM",t) ==1 Then /* comment it out */
      /*s[0]='*';*/
      category='P'; 
    ElsIf ci_prefix(".SUBCKT",t) ==1 Then /* split off any "params" tail */ 
      a= spos("PARAMS:",t);
      If a>0 Then 
        pscopy(s,s,1,a-1);
      EndIf 
      category='S';
    ElsIf ci_prefix(".CONTROL",t) ==1 Then 
      category='C'
    ElsIf ci_prefix(".ENDC",t) ==1 Then  
      category='E'
    ElsIf ci_prefix(".ENDS",t) ==1 Then  
      category='U'
    Else 
      category='.';
      n= stripbraces(s); 
      If n>0 Then category='B' EndIf  /* priority category ! */
    EndIf
  ElsIf s[0]==Intro Then /* private style preprocessor line */
    s[0]='*'; 
    category='P';
  ElsIf upcase(s[0])=='X' Then /* strip actual parameters */
    i=findsubname(dico, s);  /* i= index following last identifier in s */
    category='X';
  ElsIf s[0]=='+' Then /* continuation line */
    category='+'
  ElsIf cpos(s[0],"*$#")<=0 Then /* not a comment line! */ 
    n= stripbraces(s);
    If n>0 Then 
      category='B' /* line that uses braces */
    Else 
      category=' ' 
    EndIf; /* ordinary code line*/
  Else 
    category='*' 
  EndIf
  return category
EndFunc 

/************ core of numparam **************/

/* some day, all these nasty globals will go into the tdico structure
   and everything will get hidden behind some "handle" ...
*/

Intern int linecount= 0;  /* global: number of lines received via nupa_copy */
Intern int evalcount= 0;  /* number of lines through nupa_eval() */ 
Intern int nblog=0;       /* serial number of (debug) logfile */
Intern Bool inexpansion= False; /* flag subckt expansion phase */
Intern Bool incontrol= False;  /* flag control code sections */
Intern Bool dologfile= False; /* for debugging */
Intern Bool firstsignal=True;
Intern Pfile logfile= Null;
Intern tdico * dico=Null;

/*  already part of dico : */
/*  Str(80, srcfile);   source file */
/*  Darray(refptr, Pchar, Maxline)   pointers to source code lines */
/*  Darray(category, char, Maxline)  category of each line */

/*
   Open ouput to a log file.
   takes no action if logging is disabled.
   Open the log if not already open.
*/
Intern
Proc putlogfile(char c, int num, Pchar t)
Begin
  Strbig(Llen, u);
  Str(20,fname);
  If dologfile Then
    If(logfile == Null) Then
      scopy(fname,"logfile."); 
      Inc(nblog); nadd(fname,nblog);
      logfile=fopen(fname, "w");
    EndIf
    If(logfile != Null) Then
      cadd(u,c); nadd(u,num); 
      cadd(u,':'); cadd(u,' '); 
      sadd(u,t); cadd(u,'\n');
      fputs(u,logfile);
    EndIf
  EndIf
EndProc

Intern
Proc nupa_init( Pchar srcfile)
Begin
  int i;
  /* init the symbol table and so on, before the first  nupa_copy. */ 
  evalcount=0;
  linecount= 0;
  incontrol=False;
  placeholder= 0;
  dico= New(tdico); 
  inst_dico = New(tdico);
  initdico(dico);
  initdico(inst_dico);
  For i=0; i<Maxline; Inc(i) Do
    dico->refptr[i]= Null; 
    dico->category[i]='?';
  Done
  Sini(dico->srcfile);
  If srcfile != Null Then scopy(dico->srcfile, srcfile) EndIf
EndProc

Intern
Proc nupa_done(void)
Begin
  int i;
  Str(80,rep);
  int dictsize, nerrors;
  If logfile != Null Then
    fclose(logfile); 
    logfile=Null; 
  EndIf
  nerrors= dico->errcount;
  dictsize= donedico(dico);
  For i=Maxline-1; i>=0; Dec(i) Do
    Dispose( dico->refptr[i])
  Done 
  Dispose(dico);
  dico= Null;
  If NotZ(nerrors) Then
    /* debug: ask if spice run really wanted */
    scopy(rep," Copies=");      nadd(rep,linecount);
    sadd(rep," Evals=");        nadd(rep,evalcount);
    sadd(rep," Placeholders="); nadd(rep,placeholder);
    sadd(rep," Symbols=");      nadd(rep,dictsize); 
    sadd(rep," Errors=");       nadd(rep,nerrors); 
    cadd(rep,'\n'); ws(rep);	     
    ws("Numparam expansion errors: Run Spice anyway? y/n ? \n"); 
    rs(rep);
    If upcase(rep[0]) != 'Y' Then exit(-1) EndIf
  EndIf
  linecount= 0; 
  evalcount= 0; 
  placeholder= 0;
  /* release symbol table data */
EndProc
	     
/* SJB - Scan the line for subcircuits */
Proc nupa_scan(Pchar s, int linenum, int is_subckt)
Begin

  if ( is_subckt ) defsubckt( dico, s, linenum, 'U' );
  else             defsubckt( dico, s, linenum, 'O' );

EndProc

static char*
lower_str( char *str ) {
  char *s;

  for ( s = str; *s; s++ ) *s = tolower(*s);

  return str;
}

static char*
upper_str( char *str ) {
  char *s;

  for ( s = str; *s; s++ ) *s = toupper(*s);

  return str;
}

void
nupa_list_params(FILE *cp_out) {
  char *name;
  int  i;

  fprintf( cp_out, "\n\n" );
  for ( i = 1; i <= dico->nbd+1; i++ ) {
    if ( dico->dat[i].tp == 'R' ) {
      name = lower_str( strdup( dico->dat[i].nom ) );
      fprintf( cp_out, "       ---> %s = %g\n", name, dico->dat[i].vl );
      txfree(name);
    }
  }
}

double
nupa_get_param( char *param_name, int *found ) {
  char   *name  = upper_str(strdup(param_name));
  double result = 0;
  int    i;

  *found = 0;

  for ( i = 1; i <= dico->nbd+1; i++ ) {
    if ( strcmp( dico->dat[i].nom, name ) == 0 ) {
      result = dico->dat[i].vl;
      *found = 1;
      break;
    }
  }

  txfree(name);
  return result;
}

void
nupa_add_param( char *param_name, double value ) {
  char *up_name = upper_str( strdup( param_name ) );
  int  i        = attrib( dico, up_name, 'N' );

  dico->dat[i].vl     = value;
  dico->dat[i].tp     = 'R';
  dico->dat[i].ivl    = 0;
  dico->dat[i].sbbase = NULL;

  txfree(up_name);
}

void
nupa_add_inst_param( char *param_name, double value ) {
  char *up_name = upper_str( strdup( param_name ) );
  int  i        = attrib( inst_dico, up_name, 'N' );

  inst_dico->dat[i].vl     = value;
  inst_dico->dat[i].tp     = 'R';
  inst_dico->dat[i].ivl    = 0;
  inst_dico->dat[i].sbbase = NULL;

  txfree( up_name );
}

void
nupa_copy_inst_dico() {
  int i;

  for ( i = 1; i <= inst_dico->nbd; i++ ) {
    nupa_add_param( inst_dico->dat[i].nom, inst_dico->dat[i].vl );
  }
}

Func Pchar nupa_copy(Pchar s, int linenum)
/* returns a copy (not quite) of s in freshly allocated memory.
   linenum, for info only, is the source line number. 
   origin pointer s is kept, memory is freed later in nupa_done.
  must abort all Spice if malloc() fails.  
  Is called for the first time sequentially for all spice deck lines.
  Is then called again for all X invocation lines, top-down for
    subckts defined at the outer level, but bottom-up for local
    subcircuit expansion, but has no effect in that phase.    
  we steal a copy of the source line pointer.
  - comment-out a .param or & line
  - substitute placeholders for all {..} --> 10-digit numeric values.
*/
Begin
  Strbig(Llen,u);
  Strbig(Llen,keywd);
  Pchar t;
  int ls; 
  char c,d; 

  ls= length(s);
  While (ls>0) And (s[ls-1]<=' ') Do Dec(ls) Done
  pscopy(u,s, 1,ls); /* strip trailing space, CrLf and so on */
  dico->srcline= linenum;
  If (Not inexpansion) And (linenum >=0) And (linenum<Maxline) Then 
    Inc(linecount);
    dico->refptr[linenum]= s; 
    c= transform(dico,  u, incontrol, keywd);  
    If c=='C' Then 
      incontrol=True
    ElsIf c=='E' Then 
      incontrol=False
    EndIf
    If incontrol Then c='C' EndIf /* force it */
    d= dico->category[linenum]; /* warning if already some strategic line! */
    If (d=='P') Or (d=='S') Or (d=='X') Then
      fprintf(stderr," Numparam warning: overwriting P,S or X line (linenum == %d).\n", linenum);
    EndIf
    dico->category[linenum]= c;
  EndIf /* keep a local copy and mangle the string */
  ls=length(u);
  t = strdup(u);
  If t==NULL Then
    fputs("Fatal: String malloc crash in nupa_copy()\n", stderr);
    exit(-1)
  Else
    If Not inexpansion Then 
      putlogfile(dico->category[linenum],linenum,t) 
    EndIf;
  EndIf
  return t
EndFunc

Func int nupa_eval(Pchar s, int linenum)
/* s points to a partially transformed line.
   compute variables if linenum points to a & or .param line.
   If the original is an X line,  compute actual params.
   Else  substitute any &(expr) with the current values.
   All the X lines are preserved (commented out) in the expanded circuit.
*/
Begin
   int idef; /* subckt definition line */
   char c, keep, *ptr;
   int i;
   Str(80,subname);
   Bool err = True;

   dico->srcline= linenum;
   c= dico->category[linenum];
#ifdef TRACE_NUMPARAMS
   fprintf(stderr,"** SJB - in nupa_eval()\n");
   fprintf(stderr,"** SJB - processing line %3d: %s\n",linenum,s);	
   fprintf(stderr,"** SJB - category '%c'\n",c);
#endif /* TRACE_NUMPARAMS */	     
   If c=='P' Then /* evaluate parameters */
     nupa_assignment( dico, dico->refptr[linenum] , 'N');
   ElsIf c=='B' Then /* substitute braces line */
     err = nupa_substitute( dico, dico->refptr[linenum], s, False);
   ElsIf c=='X' Then /* compute args of subcircuit, if required */
     ptr = s;
     while ( !isspace(*ptr) ) ptr++; keep = *ptr; *ptr = '\0';
     nupa_inst_name = strdup(s); *nupa_inst_name = 'x'; *ptr = keep;
     for ( i = 0; i < strlen(nupa_inst_name); i++ ) nupa_inst_name[i] = toupper(nupa_inst_name[i]);

     idef = findsubckt( dico, s, subname);  
     If idef>0 Then
       nupa_subcktcall( dico, 
         dico->refptr[idef], dico->refptr[linenum], False);
     Else 
       putlogfile('?',linenum, "  illegal subckt call.");
     EndIf
   ElsIf c=='U' Then /*  release local symbols = parameters */
     nupa_subcktexit( dico);
   EndIf
   putlogfile('e',linenum,s);
   Inc(evalcount);
#ifdef TRACE_NUMPARAMS
   fprintf(stderr,"** SJB - leaving nupa_eval(): %s   %d\n", s, err);
   ws("** SJB -                  --> "); ws(s); wln();
   ws("** SJB - leaving nupa_eval()"); wln(); wln();
#endif /* TRACE_NUMPARAMS */
  if ( err ) return 0;
  else  return 1;
EndFunc

Func int nupa_signal(int sig, Pchar info)
/* warning: deckcopy may come inside a recursion ! substart no! */
/* info is context-dependent string data */
Begin
  putlogfile('!',sig, " Nupa Signal");
  If    sig == NUPADECKCOPY Then
    If firstsignal Then 
      nupa_init(info); 
      firstsignal=False; 
    EndIf
  ElsIf sig == NUPASUBSTART Then 
    inexpansion=True
  ElsIf sig == NUPASUBDONE  Then 
    inexpansion=False;
    nupa_inst_name = NULL;
  ElsIf sig == NUPAEVALDONE Then 
    nupa_done(); 
    firstsignal=True 
  EndIf
  return 1
EndFunc

#ifdef USING_NUPATEST
/* This is use only by the nupatest program */
Func tdico * nupa_fetchinstance(void)
Begin
  return dico
EndFunc
#endif /* USING_NUPATEST */
