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

/* Length of "numparam____ ..." string to be inserted and replaced. */

#define ACT_CHARACTS 25
#define MARKER "numparm__________"

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
    char* cardline;             /* line of card treated currently */
    const char* cardsource;     /* linesource of card treated currently
                                 * (read-only; owned by the card) — used
                                 * by HSPICE table_param() to resolve
                                 * relative .table paths against the
                                 * directory of the .lib that emitted
                                 * the call */
    bool suppress_errors;       /* when true, message() is a no-op */
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

/* Evaluate a brace-body expression in a transient scope: push
 * the (name, value) pairs, run numparam's formula(), pop the scope.
 * No persistent side effects.  Returns 0 / writes result to *out on
 * success, non-zero on error.  See xpressn.c for full docs. */
int nupa_eval_with_scope(dico_t *dico, const char *expr,
                         const char *const *names, const double *values,
                         int n, double *out);

/* External accessor for the active dico_t.  Used by callers outside
 * numparam (e.g. osdi_defer.c) that need to invoke
 * nupa_eval_with_scope() without seeing the static dicoS. */
dico_t *nupa_get_dico(void);

/* Mark a line as inert from numparam's perspective — used when a
 * later pass (e.g. inp_subcktexpand's X-prefix→OSDI fallback)
 * comments out a line that numparam had already categorized as a
 * subckt call or .param.  Without this, numparam still tries to
 * dispatch the now-commented-out line and emits "illegal subckt
 * call" / "Cannot find subcircuit" errors.  Also rewrites the
 * stored dynrefptr line to start with `*` so any remaining
 * downstream pass treats it as comment. */
void nupa_skip_line(int linenum);

/* Public-name lookup for a real-valued numparam entry.  Returns 1 and
 * sets *value on success; returns 0 if name is unknown or not a
 * NUPA_REAL.  Used by spicelib/parser/inpdpar.c so HSPICE-style bare
 * `.param` identifiers as device value fields (e.g. `vgs n2 n3 vgswp`
 * paired with `.param vgswp=0`) resolve to a leading numeric value
 * instead of erroring with "unknown parameter (vgswp)". */
int nupa_get_real(const char *name, double *value);

/* Companion writer to nupa_get_real.  Updates the value of an
 * existing NUPA_REAL entry.  Returns 1 on success, 0 if the name
 * is unknown or not NUPA_REAL.  Used by the .dc analysis loop
 * (src/spicelib/analysis/dctrcurv.c) when sweeping a .param: each
 * step updates the dictionary value, then pushes it to any V/I
 * source that bound this .param at parse time (via the
 * dpar_param_bindings table — see src/spicelib/parser/inpdpar.c). */
int nupa_set_real(const char *name, double value);
