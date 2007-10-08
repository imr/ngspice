/*
 * numparam.h 
 * $Id$
 */

/*** interface to spice frontend  subckt.c ***/

#include "numpaif.h"

/***** numparam internals ********/

#define ln(x) log(x)
#define trunc(x) floor(x)

Cconst(Nul, 0)
Cconst(Nodekey,'#') /*introduces node symbol*/
Cconst(Intro  ,'&') /*introduces preprocessor tokens*/
Cconst(Comment,'*') /*Spice Comment lines*/
Cconst(Pspice,'{')  /*Pspice expression */
Cconst(Maxdico,40000) /*size of symbol table*/

/* Composite line length
   This used to be 250 characters, but this is too easy to exceed with a
   .model line, especially when spread over several continuation 
   lines with much white space.  I hope 1000 will be enough. */
Cconst(Llen,15000)

typedef char str50 [54];
typedef char str80 [84];

Cconst(Maxline, 40000) /* size of initial unexpanded circuit code */
Cconst(Maxckt,  40000)  /* size of expanded circuit code */


typedef Pchar auxtable; /* dummy */

Record(entry)
  char   tp; /* type:  I)nt R)eal S)tring F)unction M)acro P)ointer */
  char   nom[100];
  int  level; /* subckt nesting level */
  double vl;    /* float value if defined */
  Word   ivl;   /*int value or string buffer index*/
  Pchar  sbbase; /* string buffer base address if any */
EndRec(entry)

Record(fumas) /*function,macro,string*/
   Word   start /*,stop*/ ; /*buffer index or location */
EndRec(fumas)

Record(tdico)
/* the input scanner data structure */
  str80   srcfile; /* last piece of source file name */
  int   srcline;
  int   errcount;
  entry   dat[Maxdico+1];
  int     nbd;   /* number of data entries */
  fumas   fms[101];
  int   nfms;   /* number of functions & macros */
  int   stack[20];
  char    *inst_name[20];
  int   tos;    /* top of stack index for symbol mark/release mechanics */
  str80   option; /* one-character translator options */
  auxtable nodetab;
  Darray(refptr,  Pchar, Maxline)  /* pointers to source code lines */
  Darray(category, char, Maxline) /* category of each line */
EndRec(tdico)

Proc initdico(tdico * dico);
Func int donedico(tdico * dico);
Func Bool defsubckt( tdico *dico, Pchar s, Word w, char categ);
Func int findsubckt( tdico *dico, Pchar s, Pchar subname);  
Func Bool nupa_substitute( tdico *dico, Pchar s, Pchar r, Bool err);
Func Bool nupa_assignment( tdico *dico, Pchar  s, char mode);
Func Bool nupa_subcktcall( tdico *dico, Pchar s, Pchar x, Bool err);
Proc nupa_subcktexit( tdico *dico);
Func tdico * nupa_fetchinstance(void);
Func char getidtype( tdico *d, Pchar s);
Func int attrib( tdico *dico, Pchar t, char op );

char *nupa_inst_name;
tdico *inst_dico;
