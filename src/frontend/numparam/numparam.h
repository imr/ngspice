/*
 * numparam.h
 */

/*** interface to spice frontend  subckt.c ***/

#include "numpaif.h"
#include "ngspice/hash.h"

/***** numparam internals ********/

/* -----------------------------------------------------------------
 * I believe the entry_t should be a union of type but I need more info.
 * ----------------------------------------------------------------- */

struct nupa_type;

extern const struct nupa_type S_nupa_real;
extern const struct nupa_type S_nupa_string;
extern const struct nupa_type S_nupa_subckt;
extern const struct nupa_type S_nupa_unknown;

#define  NUPA_REAL     (&S_nupa_real)
#define  NUPA_STRING   (&S_nupa_string)
#define  NUPA_SUBCKT   (&S_nupa_subckt)
#define  NUPA_UNKNOWN  (&S_nupa_unknown)

typedef const struct nupa_type *nupa_type;


typedef struct entry_s {
    nupa_type tp;      /* type: I)nt R)eal S)tring F)unction M)acro P)ointer */
    char *symbol;
    int  level;                 /* subckt nesting level */
    double vl;                  /* float value if defined */
    int  ivl;                   /* int value or string buffer index */
    char *sbbase;               /* string buffer base address if any */
} entry_t;


typedef struct {                /* the input scanner data structure */
    int srcline;
    int oldline;
    int errcount;
    int max_stack_depth;        /* alloced maximum depth of the symbol stack */
    int stack_depth;            /* current depth of the symbol stack */
    NGHASHPTR *symbols;         /* stack of scopes for symbol lookup */
                                /*  [0] denotes global scope */
    NGHASHPTR inst_symbols;     /* instance qualified symbols - after a pop */
    char **inst_name;           /* name of subcircuit */
    char **dynrefptr;
    char *dyncategory;
    int hs_compatibility;       /* allow extra keywords */
    int linecount;              /* number of lines in deck */
} dico_t;


void initdico(dico_t *);
int donedico(dico_t *);
void dico_free_entry(entry_t *);
bool defsubckt(dico_t *, const struct card *);
int findsubckt(dico_t *, const char *s);
bool nupa_substitute(dico_t *, const char *s, char **lp);
bool nupa_assignment(dico_t *, const char *s, char mode);
bool nupa_subcktcall(dico_t *, const char *s, const char *x,
        char *inst_name);
void nupa_subcktexit(dico_t *);
entry_t *entrynb(dico_t *dico, char *s);
entry_t *attrib(dico_t *, NGHASHPTR htable, char *t, char op);
void del_attrib(void *);
void nupa_copy_inst_entry(char *param_name, entry_t *proto);
