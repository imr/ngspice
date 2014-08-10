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

Darray(buff, char *, Maxline) /* input lines */
Darray(buf2, char *, Maxline) /* stripped lines */
Darray(pxbuf, char *, Maxline) /* prefix for subnodes */
Darray(runbuf, int, Maxckt) /* index list of expanded circuit */
Darray(pindex, int, Maxckt) /* prefix index list  */
int irunbuf= 0; /* count lines of runbuf */
int ipx=0;      /* count prefixes in pxbuf */  

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

 int runscript( dico_t *dico, char * prefix,
   int istart, int istop, int maxnest)
/* recursive top-down expansion: circuit --> list of line numbers */
/* keep it simple,stupid  compared to Spice's code */
/* prefix: inherited string for node & device prefixing */
/* istart, istop: allowed interval in table buf[], buf2[]. */
/* return value: number of lines included */
{
  int i,j, idef, nnest, nline, dn, myipx;
  Strbig(Llen, subpfx); /* subckt prefix */ 
  Str(80, subname);
  char c;
  unsigned char done= 0;
  i=istart; 
  nline=0;
  Inc(ipx); myipx= ipx; /* local copy */
  pxbuf[ipx]= newstring( length(prefix));
  scopy( pxbuf[ipx], prefix);
  while ( (maxnest>0) && (i<istop) && (! done) ) {
    c= dico->category[i];
    if ( c=='U' ) { 
      done=1; /* subcircuit end. Keep as a comment? */
      buf2[i][0]='#';
    }
    if ( c=='S' ) {        /* skip nested subcircuits */
      nnest=1;
      do { 
        Inc(i); c= dico->category[i];
        if ( c=='S' ) { 
          Inc(nnest); 
        } else if ( c=='U' ) { 
          Dec(nnest); 
        }
      } while ( !( (nnest<=0) || (i>=istop) ));
    } else if ( c=='X' ) {                   /* recursion here ! */
      runbuf[irunbuf]= i;
      pindex[irunbuf]= myipx;
      Inc(irunbuf); Inc(nline);
      /* keep out-commented X line for parameter passing */
      idef = findsubckt( dico, buf2[i], subname); 
      buf2[i][0]= '*'; 
      if ( idef>0 ) {
        scopy(subpfx, prefix); 
        cadd(subpfx, pfxsep);
        j=1; /* add the instance name from buf2[i] */
        while ( buf2[i][j] > ' ' ) {
          cadd( subpfx, buf2[i][j]); Inc(j);
        } 
        dn= runscript(dico, subpfx, idef+1, istop, maxnest-1);
        nline= nline+dn; 
      } else { /* FIXME: error message here! */
        printf("cannot find subckt %s\n", buf2[i]);
      }
    } else if ( (c != '?') && NotZ(buf2[i][0]) ) {         
      /*  keep any other valid non-empty line, and its prefix pointer */
      runbuf[irunbuf]= i;
      pindex[irunbuf]= myipx;
      Inc(irunbuf); Inc(nline);
    } 
    Inc(i);
  }  
  return nline;
}

void gluepluslines( int imax)
/* general sweep to eliminate continuation lines */
{
  int i,j,k, ls, p;
  Strbig(Llen,s);
  i=1;
  while ( i<= imax ) {
    if ( (buff[i][0]=='+') && (i>1) ) { 
      j= i-1; 
      while ( (i < imax) && (buff[i+1][0]=='+') ) { Inc(i) ;}
      /* the lines j+1 ... i are continuation lines to j */
      for ( k=j; k<=i; Inc(k) ) { 
        ls=length(s);
        sadd(s, buff[k]);
        p= spos("//",s);
        if ( p>0 ) { pscopy(s,s, 1,p-1) ;}
        if ( ls>0 ) { s[ls]=' ' ;} /* erase the + */;
      }
      ls= length(s);
      if ( ls> 80 ) {
        Dispose(buff[j]);
        buff[j]=newstring(ls);
      }
      scopy(buff[j], s);
    }  
    Inc(i);
  }
}

void fwrites(FILE * f, char * s)
{ 
  fputs(s,f); 
}

void fwriteln(FILE * f)
{ 
  fputc('\n',f); 
}

static
void freadln(FILE * f, char * s, int max)
{ 
  int ls;
  freadstr(f,s,max); 
  ls=length(s);
  if ( feof(f) && (ls>0) ) { 
    pscopy(s,s,1,ls-1); 
  } /* kill EOF character */;
}

void wordinsert(char * s, char * w, int i)
/* insert w before s[i] */
{
  Strbig(Llen,t);
  int ls=length(s);
  pscopy(t,s,i+1,ls); pscopy(s,s,1,i);
  sadd(s,w); sadd(s,t);
}

 int worddelete(char * s, int i)
/* delete word starting at s[i] */
{
  Strbig(Llen,t);
  int ls= length(s);
  int j=i;
  while ( (j<ls) && (s[j]>' ') ) { Inc(j) ;}
  pscopy(t,s,j+1,ls);
  pscopy(s,s,1,i); 
  sadd(s,t);
  return j-i /* nb of chars deleted */;
}

 int getnextword(char * s, char * u, int j)
{
  int ls,k;
  ls= length(s);
  k=j;
  while ( (j<ls) && (s[j] >  ' ') ) { Inc(j) ;} /* skip current word */ 
  pscopy(u, s,  k+1, j-k);
  while ( (j<ls) && (s[j] <= ' ') ) { Inc(j) ;}
  return j;
}

 int inwordlist(char * u, char * wl)
/* suppose wl is single-space separated, plus 1 space at start and end. */ 
{
  int n,p,k;
  Str(80,t);
  n=0;
  ccopy(t,' '); sadd(t,u); cadd(t,' ');
  p= spos(t,wl);
  if ( p>0 ) {
    for ( k=0; k<p; Inc(k) ) {
      if ( wl[k] <= ' ' ) { Inc(n) ;}
    }
  }
  return n;
}

void takewordlist(char * u, int k, char * wl)
{
  int i,j,lwl;
  lwl= length(wl);
  i=0; j=0;
  scopy(u,"");
  while ( (i<lwl) && (j<k ) ) {
    if ( wl[i] <= ' ' ) { Inc(j) ;}
    Inc(i);
  }
  if ( j==k ) { /* word has been found and starts at i */
    while ( wl[i]>' ' ) {
      cadd(u,wl[i]); Inc(i);
    }
  }
}

char * deviceletter= "RLCVIBSGETOUWFHDQKJZM";
char * nbofnodes   = "222222444443222240334";
char * nbsubdevice = "000000000000111002000";

void prefixing(char * s, char * p, char * formals, char * actuals,
   char categ, dico_t *dic)
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
{
  int i,j,k,ls, jnext, dsize;
  int dtype, nodes, subdv;
  unsigned char done;
  char leadchar;
  Str(80,u); Str(80,v); Str(80,pfx);
  i=0; ls=length(s);
  while ( (i<ls) && (s[i]<=' ') ) { Inc(i) ;}
  if ( alfa(s[i]) || (categ=='X') ) { /* splice in the prefix and nodelist */
    wordinsert(s,p, i+1);
    j= getnextword(s,u,i); 
    done=0;
    if ( p[0]== pfxsep ) {
      pscopy(pfx,p, 2, length(p)); 
    } else { 
      scopy(pfx,p);
    }
    leadchar=upcase(s[i]); 
    dtype= cpos( leadchar, deviceletter) -1 ;
    if ( dtype >= 0 ) { 
      nodes= nbofnodes[dtype] - '0';
      subdv= nbsubdevice[dtype] - '0';
    } else {
      nodes=999; subdv=0;
    }
    while ( ! done ) {
      jnext= getnextword(s,u,j);
      done=(jnext >= length(s)); /* was the last one, do not transform */
       /* bug: are there semilocal nodes ? in nested subckt declarations ? */
      if ( (leadchar=='Q') && (! done) ) { /* BJT: watch non-node name */
        scopy(v,u); stupcase(v);
        done=  getidtype(dic, v) == 'O'; /* a model name stops the node list */;
      } 
      if ( (! done) && (nodes>0) ) { /* transform a node name */
        k= inwordlist(u, formals);
        if ( (k>0) ) {   /* parameter node */
          dsize= - worddelete(s,j);
          takewordlist(u,k, actuals);
          wordinsert(s,u,j); 
          dsize= dsize + length(u);
        } else if ( stne(u,"0") ) {  /* local node */
          wordinsert(s,pfx,j);
          dsize= length(pfx);
        } else { dsize=0 ;}
      } else if ( (! done) && (subdv >0) ) { /* splice a subdevice name */
        wordinsert(s,p,j+1);
        dsize= length(p);
      }
      j= jnext + dsize; /*  jnext did shift ...*/
      if ( nodes >0 ) { Dec(nodes);
      } else if ( subdv >0 ) { Dec(subdv);
      }
      done= done || (Zero(nodes) && Zero(subdv));
    }
  }
}

void getnodelist(char * form, char * act, char * s, dico_t *dic, int k)
/* the line s contains the actual node parameters, between 1st & last word */
{
  int j,ls, idef;
  Str(80,u); Strbig(Llen,t);
  ccopy(act,' '); ccopy(form,' ');
  j=0; ls= length(s);
  j= getnextword(s,u,j);
  while ( j<ls ) {
    j= getnextword(s,u,j); 
    if ( j<ls ) { sadd(act,u); cadd(act,' ') ;}
  }
  /* now u already holds the subckt name if all is ok ? */
  idef = findsubckt( dic, buf2[k], u); 
  /* line buf2[idef] contains: .subckt name < formal list > */ 
  if ( idef>0 ) { 
    scopy(t, buf2[idef]); 
  } else {
    printf("Subckt call error: %s\n", s);
  }
  j=0; ls= length(t);
  j= getnextword(t,u,j); 
  j= getnextword(t,u,j); 
  while ( j<ls ) {
    j= getnextword(t,u,j); 
    sadd(form,u); cadd(form,' ');
  }
}

void nupa_test(char * fname, char mode)
/* debugging circuit expansion run. mode='w': write ouput file */
/* bugs in nupa_eval(), and for nested subckt definitions !?! */
{
  FILE * tf, fout;
  dico_t * dic; /* dictionary data pointer */
  Strbig(Llen,s);
  Str(80, prefix);
  /* Strbig(Llen, formals); Strbig(Llen,actuals); */
  Darray(formals, char *, 10)
  Darray(actuals, char *, 10)
  int i, j, k, nline, parstack;
  for ( i=0; i<Maxline; Inc(i) ) { /* allocate string storage */
    buff[i]= newstring(80);
    buf2[i]= NULL;
    pxbuf[i]= NULL; 
  }
  for ( i=0; i<10; Inc(i) ) {
    formals[i]= newstring(250);
    actuals[i]= newstring(250);
  }
  i=0; parstack=0;
  tf=fopen( fname, "r");
  if ( tf != NULL ) {
    while ( (! feof(tf)) && ((i+1) < Maxline) ) {
      Inc(i);
      freadln(tf, buff[i], 80); /* original data */;
    }
    fclose(tf);
  } else {
    printf("Cannot find %s\n", fname);
  }
  /* continuation lines are glued at this stage, so they can be ignored
     in all the subsequent manipulations.
  */
  gluepluslines(i); /* must re-allocate certain buff[i]  */ 
  nupa_signal(NUPADECKCOPY, fname);
  dic= nupa_fetchinstance(); /* bug: should have a task handle as arg */
  for ( j=1; j<=i; Inc(j) ) {
    buf2[j]= nupa_copy(buff[j], j); /* transformed data */;
  }
  nupa_signal(NUPASUBDONE, NULL);
  nline= runscript(dic, "", 1,i, 20); /* our own subckt expansion */
  /* putlogfile(' ',nline," expanded lines"); */
  if ( mode=='w' ) {
    i= cpos('.', fname); 
    pscopy(s, fname, 1, i);
    sadd(s,"out");
    fout= fopen(s, "w");
  } else {
    fout= NULL;
  }
  for ( j=0; j<irunbuf; Inc(j) ) {
    k= runbuf[j]; 
    if ( buf2[k] != NULL ) { 
      scopy(s, buf2[k]);
      nupa_eval(s, k);
      scopy(prefix,pxbuf[pindex[j]]); 
      if ( NotZ(prefix[0]) ) { cadd(prefix, pfxsep) ;} 
      prefixing(s, prefix, formals[parstack], actuals[parstack],
           dic->category[k], dic);  
      if ( dic->category[k] == 'X' ) { 
        if ( parstack< (10-1) ) { Inc(parstack) ;}
        getnodelist(formals[parstack], actuals[parstack], s, dic,k);
        /*dbg: printf("Form: %s\n", formals[parstack]); */
        /*dbg: printf("Actu: %s\n", actuals[parstack]); */;
      } else if ( dic->category[k]=='U' ) { /* return from subckt */
        if ( parstack>0 ) { Dec(parstack) ;}
      }
      if ( fout != NULL ) {
        fwrites(fout, s); fwriteln(fout);
      }
    }
  } 
  if ( fout != NULL ) { fclose(fout) ;}
  nupa_signal(NUPAEVALDONE, NULL); /* frees the buff[i] */
  for ( i= 10-1; i>=0; Dec(i) ) {
    Dispose(actuals[i]);
    Dispose(formals[i]);
  }
  for ( i= Maxline -1; i>=0; Dec(i) ) {
    Dispose(pxbuf[i]);  
    Dispose(buf2[i]); 
    /* Dispose(buff[i]) done elsewhere */;
  }
}

 int main(int argc, char * argv[])
{
  Str(80,fname);
  if ( argc>1 ) { 
    scopy(fname, argv[1]);
  } else { 
    scopy(fname,"testfile.nup"); 
  }
  nupa_test(fname, 'w');
  return 0;
}

 
