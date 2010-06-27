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
typedef enum {Defd=15} _nDefd; /* serial numb. of 'defined' keyword. The others are not used (yet) */

typedef char * auxtable; /* dummy */

/* -----------------------------------------------------------------
 * I believe the entry should be a union of type but I need more info.
 * ----------------------------------------------------------------- */
typedef struct _tentry {
  char   tp; /* type:  I)nt R)eal S)tring F)unction M)acro P)ointer */
  char *symbol ;
  int  level; /* subckt nesting level */
  double vl;    /* float value if defined */
  unsigned  ivl;   /*int value or string buffer index*/
  char *  sbbase; /* string buffer base address if any */
  struct _tentry *pointer ;	/* pointer chain */
} entry;

typedef struct _tfumas { /*function,macro,string*/
   unsigned start /*,stop*/ ; /*buffer index or location */
} fumas;

typedef struct _ttdico {
/* the input scanner data structure */
  SPICE_DSTRING srcfile; 	/* last piece of source file name */
  SPICE_DSTRING option;  	/* one-character translator options */
  SPICE_DSTRING lookup_buf ;  	/* useful temp buffer for quick symbol lookup */
  int   srcline;
  int oldline;
  int   errcount;
  int symbol_stack_alloc ;	/* stack allocation */
  int stack_depth ;		/* current depth of the symbol stack */
  NGHASHPTR global_symbols ;	/* hash table of globally defined symbols for quick lookup */
  NGHASHPTR *local_symbols ;	/* stack of locally defined symbols */
  NGHASHPTR inst_symbols ;	/* instance qualified symbols - after a pop */
  char **inst_name ;		/* name of subcircuit */
  fumas   fms[101];
  int   nfms;   /* number of functions & macros */
  auxtable nodetab;
  char **dynrefptr;
  char *dyncategory;
  int hspice_compatibility;	/* allow hspice keywords */
} tdico;

void initdico(tdico * dico);
int donedico(tdico * dico);
void dico_free_entry( entry *entry_p ) ;
unsigned char defsubckt( tdico *dico, char * s, int w, char categ);
int findsubckt( tdico *dico, char * s, SPICE_DSTRINGPTR subname);  
unsigned char nupa_substitute( tdico *dico, char * s, char * r, unsigned char err);
unsigned char nupa_assignment( tdico *dico, char *  s, char mode);
unsigned char nupa_subcktcall( tdico *dico, char * s, char * x, unsigned char err);
void nupa_subcktexit( tdico *dico);
tdico * nupa_fetchinstance(void);
char getidtype( tdico *d, char * s);
entry *attrib( tdico *d, NGHASHPTR htable, char * t, char op );
