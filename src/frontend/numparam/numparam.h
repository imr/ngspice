/*
 * numparam.h 
 * $Id$
 */

/*** interface to spice frontend  subckt.c ***/

#include "numpaif.h"
#include "hash.h"

/***** numparam internals ********/

#define ln(x) log(x)
#define trunc(x) floor(x)

typedef enum {Nul=0} _nNul;
typedef enum {Nodekey='#'} _nNodekey;   /* Introduces node symbol */
typedef enum {Intro='&'} _nIntro;       /* Introduces preprocessor tokens */
typedef enum {Comment='*'} _nComment;   /* Spice Comment lines*/
typedef enum {Pspice='{'} _nPspice;     /* Pspice expression */
//typedef enum {Maxdico=40000} _nMaxdico; /* Size of symbol table*/
typedef enum {Defd=15} _nDefd; /* serial numb. of 'defined' keyword. The others are not used (yet) */

/* Composite line length
   This used to be 250 characters, but this is too easy to exceed with a
   .model line, especially when spread over several continuation 
   lines with much white space.  Set to 40000 to catch really big
   macros in .model lines. Will add 100k of memory compared to previous 25004*/
//typedef enum {Llen=40000} _nLlen;


//typedef enum {Maxline=70000} _nMaxline; /* Size of initial unexpanded circuit code */
//typedef enum {Maxckt=40000} _nMaxckt;   /* Size of expanded circuit code */


typedef char * auxtable; /* dummy */

/* -----------------------------------------------------------------
 * I believe the entry should be a union of type but I need more info.
 * ----------------------------------------------------------------- */
typedef struct _tentry {
  char   tp; /* type:  I)nt R)eal S)tring F)unction M)acro P)ointer */
  char *symbol ;
  int  level; /* subckt nesting level */
  double vl;    /* float value if defined */
  unsigned short   ivl;   /*int value or string buffer index*/
  char *  sbbase; /* string buffer base address if any */
  struct _tentry *pointer ;	/* pointer chain */
} entry;

typedef struct _tfumas { /*function,macro,string*/
   unsigned short   start /*,stop*/ ; /*buffer index or location */
} fumas;

typedef struct _ttdico {
/* the input scanner data structure */
  SPICE_DSTRING srcfile; 	/* last piece of source file name */
  SPICE_DSTRING option;  	/* one-character translator options */
  SPICE_DSTRING lookup_buf ;  	/* useful temp buffer for quick symbol lookup */
  int   srcline;
  int oldline;
  int   errcount;
  int   num_symbols ;		/* number of symbols in entry array */
  entry **symbol_array ;	/* symbol entries in array format for stack ops */
  NGHASHPTR symbol_table ;	/* hash table of symbols for quick lookup */
  int     nbd;                  /* number of data entries */
  fumas   fms[101];
  int   nfms;   /* number of functions & macros */
  int   stack[20];
  char    *inst_name[20];
  int   tos;    /* top of stack index for symbol mark/release mechanics */
  auxtable nodetab;
//  char * refptr[Maxline]; /* pointers to source code lines */
  char **dynrefptr;
//  char category[Maxline]; /* category of each line */
  char *dyncategory;
  int hspice_compatibility;	/* allow hspice keywords */
} tdico;

void initdico(tdico * dico);
int donedico(tdico * dico);
unsigned char defsubckt( tdico *dico, char * s, int w, char categ);
int findsubckt( tdico *dico, char * s, SPICE_DSTRINGPTR subname);  
unsigned char nupa_substitute( tdico *dico, char * s, char * r, unsigned char err);
unsigned char nupa_assignment( tdico *dico, char *  s, char mode);
unsigned char nupa_subcktcall( tdico *dico, char * s, char * x, unsigned char err);
void nupa_subcktexit( tdico *dico);
tdico * nupa_fetchinstance(void);
char getidtype( tdico *d, char * s);
entry *attrib( tdico *dico, char * t, char op );
