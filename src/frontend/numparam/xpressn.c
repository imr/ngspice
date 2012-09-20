/* xpressn.c                Copyright (C)  2002    Georg Post

   This file is part of Numparam, see:  readme.txt
   Free software under the terms of the GNU Lesser General Public License
*/

#include <stdio.h>                /* for function message() only. */
#include <stdarg.h>

#include "general.h"
#include "numparam.h"
#include "ngspice/ngspice.h"
#include "ngspice/cpdefs.h"
#include "ngspice/ftedefs.h"
#include "ngspice/dvec.h"
#include "../frontend/variable.h"
#include "ngspice/compatmode.h"


/* random numbers in /maths/misc/randnumb.c */
extern double gauss0(void);
extern double drand(void);

/************ keywords ************/

/* SJB - 150 chars is ample for this - see initkeys() */
static SPICE_DSTRING keyS;      /* all my keywords */
static SPICE_DSTRING fmathS;    /* all math functions */

extern char *nupa_inst_name;    /* see spicenum.c */
extern long dynsubst;           /* see inpcom.c */

#define MAX_STRING_INSERT 17 /* max. string length to be inserted and replaced */
#define ACT_CHARACTS 17      /* actual string length to be inserted and replaced */
#define EXP_LENGTH 5


static double
ternary_fcn(int conditional, double if_value, double else_value)
{
    if (conditional)
        return if_value;
    else
        return else_value;
}


static double
agauss(double nominal_val, double abs_variation, double sigma)
{
    double stdvar;
    stdvar = abs_variation / sigma;
    return (nominal_val + stdvar * gauss0());
}


static double
gauss(double nominal_val, double rel_variation, double sigma)
{
    double stdvar;
    stdvar = nominal_val * rel_variation / sigma;
    return (nominal_val + stdvar * gauss0());
}


static double
unif(double nominal_val, double rel_variation)
{
    return (nominal_val + nominal_val * rel_variation * drand());
}


static double
aunif(double nominal_val, double abs_variation)
{
    return (nominal_val + abs_variation * drand());
}


static double
limit(double nominal_val, double abs_variation)
{
    return (nominal_val + (drand() > 0 ? abs_variation : -1. * abs_variation));
}


static void
initkeys(void)
/* the list of reserved words */
{
    spice_dstring_init(&keyS);
    scopy_up(&keyS,
             "and or not div mod if else end while macro funct defined"
             " include for to downto is var");
    scopy_up(&fmathS,
             "sqr sqrt sin cos exp ln arctan abs pow pwr max min int log sinh cosh"
             " tanh ternary_fcn v agauss sgn gauss unif aunif limit ceil floor");
}


static double
mathfunction(int f, double z, double x)
/* the list of built-in functions. Patch 'fmath', here and near line 888 to get more ...*/
{
    double y;
    switch (f)
    {
    case 1:
        y = x * x;
        break;
    case 2:
        y = sqrt(x);
        break;
    case 3:
        y = sin(x);
        break;
    case 4:
        y = cos(x);
        break;
    case 5:
        y = exp(x);
        break;
    case 6:
        y = ln(x);
        break;
    case 7:
        y = atan(x);
        break;
    case 8:
        y = fabs(x);
        break;
    case 9:
        y = pow(z, x);
        break;
    case 10:
        y = exp(x * ln(fabs(z)));
        break;
    case 11:
        y = MAX(x, z);
        break;
    case 12:
        y = MIN(x, z);
        break;
    case 13:
        y = trunc(x);
        break;
    case 14:
        y = log(x);
        break;
    case 15:
        y = sinh(x);
        break;
    case 16:
        y = cosh(x);
        break;
    case 17:
        y = sinh(x)/cosh(x);
        break;
    case 21: /* sgn */
        if (x > 0)
            y = 1.;
        else if (x == 0)
            y = 0.;
        else
            y = -1.;
        break;
    case 26:
        y = ceil(x);
        break;
    case 27:
        y = floor(x);
        break;
    default:
        y = x;
        break;
    }

    return y;
}


#ifdef __GNUC__
static bool message(tdico *dic, const char *fmt, ...)
    __attribute__ ((format (__printf__, 2, 3)));
#endif


static bool
message(tdico *dic, const char *fmt, ...)
{
    va_list ap;

    char *srcfile = spice_dstring_value(&(dic->srcfile));

    if (srcfile && *srcfile)
        fprintf(stderr, "%s:", srcfile);

    if (dic->srcline >= 0)
        fprintf
            (stderr,
             "Original line no.: %d, new internal line no.: %d:\n",
             dic->oldline, dic->srcline);

    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);

    fprintf(stderr, "\n");

    dic->errcount++;

    return 1; /* error! */
}


/************ the input text symbol table (dictionary) *************/

void
initdico(tdico *dico)
{
    int asize;                          /* default allocation size */
    COMPATMODE_T compat_mode;

    spice_dstring_init(&(dico->option));
    spice_dstring_init(&(dico->srcfile));

    dico->srcline = -1;
    dico->errcount = 0;

    dico->global_symbols = nghash_init(NGHASH_MIN_SIZE);
    nghash_unique(dico->global_symbols, TRUE); /* no rewrite of global symbols */
    spice_dstring_init(&(dico->lookup_buf));

    dico->stack_depth = 0;              /* top of the stack */
    asize = dico->symbol_stack_alloc = 10;/* expected stack depth - no longer limited */
    asize++;                            /* account for zero */
    dico->local_symbols = TMALLOC(NGHASHPTR, asize);
    dico->inst_name = TMALLOC(char*, asize);
    dico->inst_symbols = NULL;          /* instance qualified are lazily allocated */

    initkeys();

    compat_mode = ngspice_compat_mode();

    if (compat_mode == COMPATMODE_HS)
        dico->hs_compatibility = 1;
    else
        dico->hs_compatibility = 0;
}


void
dico_free_entry(entry *entry_p)
{
    if (entry_p->symbol)
        txfree(entry_p->symbol);

    txfree(entry_p);
}


/* local semantics for parameters inside a subckt */
/* arguments as wll as .param expressions  */
/* to do:  scope semantics ?
   "params:" and all new symbols should have local scope inside subcircuits.
   redefinition of old symbols gives a warning message.
*/

typedef enum {Push = 'u'} _nPush;
typedef enum {Pop  = 'o'} _nPop;


static void
dicostack(tdico *dico, char op)
/* push or pop operation for nested subcircuit locals */
{
    int asize;                  /* allocation size */
    char *inst_name;            /* name of subcircuit instance */
    char *param_p;              /* qualified inst parameter name */
    entry *entry_p;             /* current entry */
    NGHASHPTR htable_p;         /* current hash table */
    NGHASHITER iter;            /* hash iterator - thread safe */

    if (op == Push) {
        dico->stack_depth++;
        if (dico->stack_depth > dico->symbol_stack_alloc) {
            /* Just double the stack alloc */
            dico->symbol_stack_alloc *= 2;
            asize = dico->symbol_stack_alloc + 1; /* account for zero */
            dico->local_symbols = TREALLOC(NGHASHPTR, dico->local_symbols, asize);
            dico->inst_name = TREALLOC(char*, dico->inst_name, asize);
        }
        /* lazy allocation - don't allocate space if we can help it */
        dico->local_symbols[dico->stack_depth] = NULL;
        dico->inst_name[dico->stack_depth] = nupa_inst_name;
    } else if (op == Pop) {
        if (dico->stack_depth > 0) {
            /* -----------------------------------------------------------------
             * Keep instance parameters around by transferring current local
             * scope variables to an instance qualified hash table.
             * ----------------------------------------------------------------- */
            inst_name = dico->inst_name[dico->stack_depth];
            htable_p = dico->local_symbols[dico->stack_depth];
            if (htable_p) {
                SPICE_DSTRING param_name; /* build a qualified name */
                spice_dstring_init(&param_name);

                NGHASH_FIRST(&iter);
                for (entry_p = (entry *) nghash_enumerateRE(htable_p, &iter);
                     entry_p;
                     entry_p = (entry *) nghash_enumerateRE(htable_p, &iter))
                {
                    spice_dstring_reinit(&param_name);
                    param_p = spice_dstring_print(&param_name, "%s.%s",
                                                  inst_name, entry_p->symbol);
                    nupa_add_inst_param(param_p, entry_p->vl);
                    dico_free_entry(entry_p);
                }
                nghash_free(htable_p, NULL, NULL);
                spice_dstring_free(&param_name);
            }
            tfree(inst_name);

            dico->inst_name[dico->stack_depth] = NULL;
            dico->local_symbols[dico->stack_depth] = NULL;
            dico->stack_depth--;

        } else {
            message(dico, " Subckt Stack underflow.");
        }
    }
}


int
donedico(tdico *dico)
{
    int sze = nghash_get_size(dico->global_symbols);
    return sze;
}


/* -----------------------------------------------------------------
 * Now entryb works on the given hash table hierarchy.   First
 * look thru the stack of local symbols and then look at the global
 * symbols in that order.
 * ----------------------------------------------------------------- */
static entry *
entrynb(tdico *d, char *s)
{
    int depth;                  /* stack depth */
    entry *entry_p;             /* search hash table */
    NGHASHPTR htable_p;         /* hash table */

    /* look at the current scope and then backup the stack */
    for (depth = d->stack_depth; depth > 0; depth--) {
        htable_p = d->local_symbols[depth];
        if (htable_p) {
            entry_p = (entry *) nghash_find(htable_p, s);
            if (entry_p)
                return (entry_p);
        }
    }

    /* No local symbols - try the global table */
    entry_p = (entry *) nghash_find(d->global_symbols, s);
    return (entry_p);
}


char
getidtype(tdico *d, char *s)
/* test if identifier s is known. Answer its type, or '?' if not in table */
{
    entry *entry_p;             /* hash table entry */
    char itp = '?';             /* assume unknown */

    entry_p = entrynb(d, s);
    if (entry_p)
        itp = entry_p->tp;

    return (itp);
}


static double
fetchnumentry(tdico *dico, char *t, bool *perr)
{
    bool err = *perr;
    double u;
    entry *entry_p;             /* hash table entry */

    entry_p = entrynb(dico, t); /* no keyword */
    /*dbg -- if (k <= 0) { printf("Dico num lookup fails."); } */

    while (entry_p && (entry_p->tp == 'P'))
        entry_p = entry_p->pointer;

    if (entry_p)
        if (entry_p->tp != 'R')
            entry_p = NULL;

    if (entry_p) {
        u = entry_p->vl;
    } else {
        err = message(dico, "Undefined number [%s]", t);
        u = 0.0;
    }

    *perr = err;

    return u;
}


/*******  writing dictionary entries *********/

entry *
attrib(tdico *dico_p, NGHASHPTR htable_p, char *t, char op)
{
    /* seek or attribute dico entry number for string t.
       Option  op='N' : force a new entry, if tos>level and old is  valid.
    */
    entry *entry_p;             /* symbol table entry */

    entry_p = (entry *) nghash_find(htable_p, t);
    if (entry_p && (op == 'N') &&
        (entry_p->level < dico_p->stack_depth) && (entry_p->tp != '?'))
    {
        entry_p = NULL;
    }

    if (!entry_p) {
        entry_p = TMALLOC(entry, 1);
        entry_p->symbol = strdup(t);
        entry_p->tp = '?';      /* signal Unknown */
        entry_p->level = dico_p->stack_depth;
        nghash_insert(htable_p, t, entry_p);
    }

    return entry_p;
}


static bool
define(tdico *dico,
       char *t,                 /* identifier to define */
       char op,                 /* option */
       char tpe,                /* type marker */
       double z,                /* float value if any */
       int w,                   /* integer value if any */
       entry *pval,             /* pointer value if any */
       char *base)              /* string pointer if any */
{
    /*define t as real or integer,
      opcode= 'N' impose a new item under local conditions.
      check for pointers, too, in full macrolanguage version:
      Call with 'N','P',0.0, ksymbol ... for VAR parameter passing.
      Overwrite warning, beware: During 1st pass (macro definition),
      we already make symbol entries which are dummy globals !
      we mark each id with its subckt level, and warn if write at higher one.
    */
    char c;
    bool err, warn;
    entry *entry_p;             /* spice table entry */
    NGHASHPTR htable_p;         /* hash table */

    NG_IGNORE(pval);

    if (dico->stack_depth > 0) {
        /* can't be lazy anymore */
        if (!(dico->local_symbols[dico->stack_depth]))
            dico->local_symbols[dico->stack_depth] = nghash_init(NGHASH_MIN_SIZE);

        htable_p = dico->local_symbols[dico->stack_depth];
    } else {
        /* global symbol */
        htable_p = dico->global_symbols;
    }

    entry_p = attrib(dico, htable_p, t, op);
    err = 0;

    if (!entry_p) {

        err = message(dico, " Symbol table overflow");

    } else {

        if (entry_p->tp == 'P')
            entry_p = entry_p->pointer; /* pointer indirection */

        if (entry_p)
            c = entry_p->tp;
        else
            c = ' ';

        if ((c == 'R') || (c == 'S') || (c == '?')) {

            entry_p->vl = z;
            entry_p->tp = tpe;
            entry_p->ivl = w;
            entry_p->sbbase = base;
            /* if ((c != '?') && (i <= dico->stack[dico->tos])) { */
            if (c == '?')
                entry_p->level = dico->stack_depth; /* promote! */

            /* warn about re-write to a global scope! */
            if (entry_p->level < dico->stack_depth)
                warn = message(dico, "%s:%d overwritten.", t, entry_p->level);

        } else {
            /* suppress error message, resulting from multiple definition of
               symbols (devices) in .model lines with same name, but in different subcircuits.
               Subcircuit expansion is o.k., we have to deal with this numparam
               behaviour later. (H. Vogt 090426)
            */
            if (0)
                message(dico, "%s: cannot redefine", t);
        }
    }

    return err;
}


bool
defsubckt(tdico *dico, char *s, int w, char categ)
/* called on 1st pass of spice source code,
   to enter subcircuit (categ=U) and model (categ=O) names
*/
{
    bool err;
    int i, j, ls;

    ls = length(s);
    i = 0;

    while ((i < ls) && (s[i] != '.'))
        i++;                    /* skip 1st dotword */

    while ((i < ls) && (s[i] > ' '))
        i++;

    while ((i < ls) && (s[i] <= ' '))
        i++;                    /* skip blank */

    j = i;

    while ((j < ls) && (s[j] > ' '))
        j++;

    if (j > i) {
        SPICE_DSTRING ustr;     /* temp user string */
        spice_dstring_init(&ustr);
        pscopy_up(&ustr, s, i, j - i);
        err = define(dico, spice_dstring_value(&ustr), ' ', categ, 0.0, w, NULL, NULL);
        spice_dstring_free(&ustr);
    } else {
        err = message(dico, "Subcircuit or Model without name.");
    }

    return err;
}


int
findsubckt(tdico *dico, char *s, SPICE_DSTRINGPTR subname)
/* input: s is a subcircuit invocation line.
   returns 0 if not found, else the stored definition line number value
   and the name in string subname  */
{
    entry *entry_p;             /* symbol table entry */
    SPICE_DSTRING ustr;         /* u= subckt name is last token in string s */
    int j, k;
    int line;                   /* stored line number */

    spice_dstring_init(&ustr);

    k = length(s);

    while ((k >= 0) && (s[k] <= ' '))
        k--;

    j = k;

    while ((k >= 0) && (s[k] > ' '))
        k--;

    pscopy_up(&ustr, s, k + 1, j - k);
    entry_p = entrynb(dico, spice_dstring_value(&ustr));

    if (entry_p && (entry_p->tp == 'U')) {
        line = entry_p->ivl;
        scopyd(subname, &ustr);
    } else {
        line = 0;
        spice_dstring_reinit(subname);
        message(dico, "Cannot find subcircuit.");
    }

    return line;
}


#if 0                           /* unused, from the full macro language... */
static int
deffuma(                        /* define function or macro entry. */
    tdico *dico, char *t, char tpe, unsigned short bufstart,
    unsigned char *pjumped, bool *perr)
{
    unsigned char jumped = *pjumped;
    bool err = *perr;
    /* if not jumped, define new function or macro, returns index to buffferstart
       if jumped, return index to existing function
    */
    int i, j;

    Strbig(Llen, v);

    i = attrib(dico, t, ' ');
    j = 0;

    if (i <= 0) {
        err = message(dico, " Symbol table overflow");
    } else {
        if (dico->dat[i].tp != '?') {
            /* old item! */
            if (jumped)
                j = dico->dat[i].ivl;
            else
                err = message(dico, "%s already defined", t);
        } else {
            dico->dat[i].tp = tpe;
            dico->nfms++;
            j = dico->nfms;
            dico->dat[i].ivl = j;
            dico->fms[j].start = bufstart;
            /* = ibf->bufaddr = start addr in buffer */
        }
    }

    *pjumped = jumped;
    *perr = err;
    return j;
}
#endif


/************ input scanner stuff **************/

static unsigned char
keyword(SPICE_DSTRINGPTR keys_p, SPICE_DSTRINGPTR tstr_p)
{
    /* return 0 if t not found in list keys, else the ordinal number */
    unsigned char i, j, k;
    int lt, lk;
    bool ok;
    char *t;
    char *keys;

    lt = spice_dstring_length(tstr_p);
    t = spice_dstring_value(tstr_p);
    lk = spice_dstring_length(keys_p);
    keys = spice_dstring_value(keys_p);
    k = 0;
    j = 0;

    do
    {
        j++;
        i = 0;
        ok = 1;

        do
        {
            i++;
            k++;
            ok = (k <= lk) && (t[i - 1] == keys[k - 1]);
        } while (ok && (i < lt));

        if (ok)
            ok = (k == lk) || (keys[k] <= ' ');

        if (!ok && (k < lk))    /* skip to next item */
            while ((k <= lk) && (keys[k - 1] > ' '))
                k++;
    } while (!ok && (k < lk));

    if (ok)
        return j;
    else
        return 0;
}


static double
parseunit(char *s)
/* the Spice suffixes */
{
    switch (toupper(s[0]))
    {
    case 'G':  return 1e9;
    case 'K':  return 1e3;
    case 'M':  return ci_prefix("MEG", s) ? 1e6 : 1e-3;
    case 'U':  return 1e-6;
    case 'N':  return 1e-9;
    case 'P':  return 1e-12;
    case 'F':  return 1e-15;
    default :  return 1;
    }
}


static int
fetchid(char *s, SPICE_DSTRINGPTR t, int ls, int i)
/* copy next identifier from s into t, advance and return scan index i */
{
    char c;
    bool ok;

    c = s[i - 1];

    while (!alfa(c) && (i < ls)) {
        i++;
        c = s[i - 1];
    }

    spice_dstring_reinit(t);
    cadd(t, upcase(c));

    do
    {
        i++;
        if (i <= ls)
            c = s[i - 1];
        else
            c = '\0';

        c = upcase(c);
        ok = alfanum(c) || c == '.';

        if (ok)
            cadd(t, c);

    } while (ok);

    return i;                   /* return updated i */
}


static double
exists(tdico *d, char *s, int *pi, bool *perror)
/* check if s in simboltable 'defined': expect (ident) and return 0 or 1 */
{
    bool error = *perror;
    int i = *pi;
    double x;
    int ls;
    char c;
    bool ok;
    SPICE_DSTRING t;

    ls = length(s);
    spice_dstring_init(&t);
    x = 0.0;

    do
    {
        i++;
        if (i > ls)
            c = '\0';
        else
            c = s[i - 1];

        ok = (c == '(');

    } while (!ok && (c != '\0'));

    if (ok)
    {
        i = fetchid(s, &t, ls, i);
        i--;
        if (entrynb(d, spice_dstring_value(&t)))
            x = 1.0;

        do
        {
            i++;

            if (i > ls)
                c = '\0';
            else
                c = s[i - 1];

            ok = (c == ')');

        } while (!ok && (c != '\0'));
    }

    if (!ok)
        error = message(d, " Defined() syntax");

    /* keep pointer on last closing ")" */

    *perror = error;
    *pi = i;
    spice_dstring_free(&t);

    return x;
}


static double
fetchnumber(tdico *dico, char *s, int *pi, bool *perror)
/* parse a Spice number in string s */
{
    double u;
    int n = 0;

    s += *pi - 1;               /* broken semantic !! */

    if (1 != sscanf(s, "%lG%n", &u, &n)) {

        *perror = message(dico, "Number format error: \"%s\"", s);

        return 0.0;             /* FIXME return NaN */

    }

    u *= parseunit(s + n);

    /* swallow unit
     *   FIXME `100MegBaz42' should emit an error message
     *   FIXME should we allow whitespace ?   `100 MEG' ?
     */

    while (s[n] && alfa(s[n]))
        n++;

    *pi += n-1;                 /* very broken semantic !!! */

    return u;
}


static char
fetchoperator(tdico *dico,
              char *s, int ls,
              int *pi,
              unsigned char *pstate, unsigned char *plevel,
              bool *perror)
/* grab an operator from string s and advance scan index pi.
   each operator has: one-char alias, precedence level, new interpreter state.
*/
{
    int i = *pi;
    unsigned char state = *pstate;
    unsigned char level = *plevel;
    bool error = *perror;
    char c, d;

    c = s[i - 1];

    if (i < ls)
        d = s[i];
    else
        d = '\0';

    if ((c == '!') && (d == '=')) {
        c = '#';
        i++;
    } else if ((c == '<') && (d == '>')) {
        c = '#';
        i++;
    } else if ((c == '<') && (d == '=')) {
        c = 'L';
        i++;
    } else if ((c == '>') && (d == '=')) {
        c = 'G';
        i++;
    } else if ((c == '*') && (d == '*')) {
        c = '^';
        i++;
    } else if ((c == '=') && (d == '=')) {
        i++;
    } else if ((c == '&') && (d == '&')) {
        i++;
    } else if ((c == '|') && (d == '|')) {
        i++;
    } if ((c == '+') || (c == '-')) {
        state = 2;              /* pending operator */
        level = 4;
    } else if ((c == '*') || (c == '/') || (c == '%') || (c == '\\')) {
        state = 2;
        level = 3;
    } else if (c == '^') {
        state = 2;
        level = 2;
    } else if (cpos(c, "=<>#GL") >= 0) {
        state = 2;
        level = 5;
    } else if (c == '&') {
        state = 2;
        level = 6;
    } else if (c == '|') {
        state = 2;
        level = 7;
    } else if (c == '!') {
        state = 3;
    } else {
        state = 0;
        if (c > ' ')
            error = message(dico, "Syntax error: letter [%c]", c);
    }

    *pi = i;
    *pstate = state;
    *plevel = level;
    *perror = error;

    return c;
}


static char
opfunctkey(tdico *dico,
           unsigned char kw, char c,
           unsigned char *pstate, unsigned char *plevel,
           bool *perror)
/* handle operator and built-in keywords */
{
    unsigned char state = *pstate;
    unsigned char level = *plevel;
    bool error = *perror;

    /*if kw operator keyword, c=token*/
    switch (kw)
    {
                                /* & | ~ DIV MOD  Defined */
    case 1:
        c = '&';
        state = 2;
        level = 6;
        break;
    case 2:
        c = '|';
        state = 2;
        level = 7;
        break;
    case 3:
        c = '!';
        state = 3;
        level = 1;
        break;
    case 4:
        c = '\\';
        state = 2;
        level = 3;
        break;
    case 5:
        c = '%';
        state = 2;
        level = 3;
        break;
    case Defd:
        c = '?';
        state = 1;
        level = 0;
        break;
    default:
        state = 0;
        error = message(dico, " Unexpected Keyword");
        break;
    }

    *pstate = state;
    *plevel = level;
    *perror = error;

    return c;
}


static double
operate(char op, double x, double y)
{
    /* execute operator op on a pair of reals */
    /* bug:   x:=x op y or simply x:=y for empty op?  No error signalling! */
    double u = 1.0;
    double z = 0.0;
    double epsi = 1e-30;
    double t;

    switch (op)
    {
    case ' ':
        x = y;                  /* problem here: do type conversions ?! */
        break;
    case '+':
        x = x + y;
        break;
    case '-':
        x = x - y;
        break;
    case '*':
        x = x * y;
        break;
    case '/':
        // if (absf(y) > epsi)
        x = x / y;
        break;
    case '^':                   /* power */
        t = absf(x);
        if (t < epsi)
            x = z;
        else
            x = exp(y * ln(t));
        break;
    case '&':                   /* && */
        if (y < x)
            x = y;              /*=Min*/
        break;
    case '|':                   /* || */
        if (y > x)
            x = y;              /*=Max*/
        break;
    case '=':
        if (x == y)
            x = u;
        else
            x = z;
        break;
    case '#':                   /* <> */
        if (x != y)
            x = u;
        else
            x = z;
        break;
    case '>':
        if (x > y)
            x = u;
        else
            x = z;
        break;
    case '<':
        if (x < y)
            x = u;
        else
            x = z;
        break;
    case 'G':                   /* >= */
        if (x >= y)
            x = u;
        else
            x = z;
        break;
    case 'L':                   /* <= */
        if (x <= y)
            x = u;
        else
            x = z;
        break;
    case '!':                   /* ! */
        if (y == z)
            x = u;
        else
            x = z;
        break;
    case '%':                   /* % */
        t = np_trunc(x / y);
        x = x - y * t;
        break;
    case '\\':                  /* / */
        x = np_trunc(absf(x / y));
        break;
    }

    return x;
}


static double
formula(tdico *dico, char *s, bool *perror)
{
    /* Expression parser.
       s is a formula with parentheses and math ops +-* / ...
       State machine and an array of accumulators handle operator precedence.
       Parentheses handled by recursion.
       Empty expression is forbidden: must find at least 1 atom.
       Syntax error if no toggle between binoperator && (unop/state1) !
       States : 1=atom, 2=binOp, 3=unOp, 4= stop-codon.
       Allowed transitions:  1->2->(3,1) and 3->(3,1).
    */
    typedef enum {nprece = 9} _nnprece; /* maximal nb of precedence levels */
    bool error = *perror;
    bool negate = 0;
    unsigned char state, oldstate, topop, ustack, level, kw, fu;
    double u = 0.0, v, w = 0.0;
    double accu[nprece + 1];
    char oper[nprece + 1];
    char uop[nprece + 1];
    int i, k, ls, natom, arg2, arg3;
    char c, d;
    bool ok;
    SPICE_DSTRING tstr;

    spice_dstring_init(&tstr);

    for (i = 0; i <= nprece; i++) {
        accu[i] = 0.0;
        oper[i] = ' ';
    }

    i = 0;
    ls = length(s);

    while ((ls > 0) && (s[ls - 1] <= ' '))
        ls--;                   /* clean s */

    state = 0;
    natom = 0;
    ustack = 0;
    topop = 0;
    oldstate = 0;
    fu = 0;
    error = 0;
    level = 0;

    while ((i < ls) && !error) {
        i++;
        c = s[i - 1];
        if (c == '(') {
            /* sub-formula or math function */
            level = 1;
            /* new: must support multi-arg functions */
            k = i;
            arg2 = 0;
            v = 1.0;
            arg3 = 0;

            do
            {
                k++;
                if (k > ls)
                    d = '\0';
                else
                    d = s[k - 1];

                if (d == '(')
                    level++;
                else if (d == ')')
                    level--;

                if ((d == ',') && (level == 1)) {
                    if (arg2 == 0)
                        arg2 = k;
                    else
                        arg3 = k; /* kludge for more than 2 args (ternary expression) */
                }                 /* comma list? */

            } while ((k <= ls) && !((d == ')') && (level <= 0)));

            if (k > ls) {
                error = message(dico, "Closing \")\" not found.");
                natom++;        /* shut up other error message */
            } else {
                if (arg2 > i) {
                    pscopy(&tstr, s, i, arg2 - i - 1);
                    v = formula(dico, spice_dstring_value(&tstr), &error);
                    i = arg2;
                }
                if (arg3 > i) {
                    pscopy(&tstr, s, i, arg3 - i - 1);
                    w = formula(dico, spice_dstring_value(&tstr), &error);
                    i = arg3;
                }
                pscopy(&tstr, s, i, k - i - 1);
                u = formula(dico, spice_dstring_value(&tstr), &error);
                state = 1;      /* atom */
                if (fu > 0) {
                    if ((fu == 18))
                        u = ternary_fcn((int) v, w, u);
                    else if ((fu == 20))
                        u = agauss(v, w, u);
                    else if ((fu == 22))
                        u = gauss(v, w, u);
                    else if ((fu == 23))
                        u = unif(v, u);
                    else if ((fu == 24))
                        u = aunif(v, u);
                    else if ((fu == 25))
                        u = limit(v, u);
                    else
                        u = mathfunction(fu, v, u);
                }
            }
            i = k;
            fu = 0;
        } else if (alfa(c)) {
            i = fetchid(s, &tstr, ls, i); /* user id, but sort out keywords */
            state = 1;
            i--;
            kw = keyword(&keyS, &tstr); /* debug ws('[',kw,']'); */
            if (kw == 0) {
                fu = keyword(&fmathS, &tstr); /* numeric function? */
                if (fu == 0)
                    u = fetchnumentry(dico, spice_dstring_value(&tstr), &error);
                else
                    state = 0;  /* state==0 means: ignore for the moment */
            } else {
                c = opfunctkey(dico, kw, c, &state, &level, &error);
            }

            if (kw == Defd)
                u = exists(dico, s, &i, &error);
        } else if (((c == '.') || ((c >= '0') && (c <= '9')))) {
            u = fetchnumber(dico, s, &i, &error);
            if (negate) {
                u = -1 * u;
                negate = 0;
            }
            state = 1;
        } else {
            c = fetchoperator(dico, s, ls, &i, &state, &level, &error);
        }

        /* may change c to some other operator char! */
        /* control chars <' '  ignored */

        ok = (oldstate == 0) || (state == 0) ||
            ((oldstate == 1) && (state == 2)) ||
            ((oldstate != 1) && (state != 2));

        if (oldstate == 2 && state == 2 && c == '-') {
            ok = 1;
            negate = 1;
            continue;
        }

        if (!ok)
            error = message(dico, " Misplaced operator");

        if (state == 3) {
            /* push unary operator */
            ustack++;
            uop[ustack] = c;
        } else if (state == 1) {
            /* atom pending */
            natom++;
            if (i >= ls) {
                state = 4;
                level = topop;
            } /* close all ops below */

            for (k = ustack; k >= 1; k--)
                u = operate(uop[k], u, u);

            ustack = 0;
            accu[0] = u;        /* done: all pending unary operators */
        }

        if ((state == 2) || (state == 4)) {
            /* do pending binaries of priority Upto "level" */
            for (k = 1; k <= level; k++) {
                /* not yet speed optimized! */
                accu[k] = operate(oper[k], accu[k], accu[k - 1]);
                accu[k - 1] = 0.0;
                oper[k] = ' ';  /* reset intermediates */
            }
            oper[level] = c;

            if (level > topop)
                topop = level;
        }

        if (state > 0)
            oldstate = state;
    }

    if ((natom == 0) || (oldstate != 4))
        error = message(dico, " Expression err: %s", s);

    if (negate == 1)
        error = message(dico,
                        " Problem with formula eval -- wrongly determined negation!");

    *perror = error;

    spice_dstring_free(&tstr);

    if (error)
        return 1.0;
    else
        return accu[topop];
}































static bool
evaluate(tdico *dico, SPICE_DSTRINGPTR qstr_p, char *t, unsigned char mode)
{
    /* transform t to result q. mode 0: expression, mode 1: simple variable */
    double u = 0.0;
    int j, lq;
    char dt;
    entry *entry_p;
    bool numeric, done, nolookup;
    bool err;

    spice_dstring_reinit(qstr_p);
    numeric = 0;
    err = 0;

    if (mode == 1) {
        /* string? */
        stupcase(t);
        entry_p = entrynb(dico, t);
        nolookup = !entry_p;

       while (entry_p && (entry_p->tp == 'P'))
            entry_p = entry_p->pointer; /* follow pointer chain */

        /* pointer chain */
        if (entry_p)
            dt = entry_p->tp;
        else
            dt = ' ';

        /* data type: Real or String */
        if (dt == 'R') {
            u = entry_p->vl;
            numeric = 1;
        } else if (dt == 'S') {
            /* suppose source text "..." at */
            j = entry_p->ivl;
            lq = 0;

            do
            {
                j++;
                lq++;
                dt = /* ibf->bf[j]; */ entry_p->sbbase[j];

                if (cpos('3', spice_dstring_value(&dico->option)) <= 0)
                    dt = upcase(dt); /* spice-2 */

                done = (dt == '\"') || (dt < ' ') || (lq > 99);

                if (!done)
                    cadd(qstr_p, dt);

            } while (!done);
        }

        if (!entry_p)
            err = message(dico,
                          "\"%s\" not evaluated.%s", t,
                          nolookup ? " Lookup failure." : "");
    } else {
        u = formula(dico, t, &err);
        numeric = 1;
    }

    if (numeric) {
        /* we want *exactly* 17 chars, we have
         *   sign, leading digit, '.', 'e', sign, upto 3 digits exponent
         * ==> 8 chars, thus we have 9 left for precision
         * don't print a leading '+', something choked
         */

        char buf[17+1];
        if (snprintf(buf, sizeof(buf), "% 17.9e", u) != 17) {
            fprintf(stderr, "ERROR: xpressn.c, %s(%d)\n", __FUNCTION__, __LINE__);
            controlled_exit(1);
        }
        scopys(qstr_p, buf);
    }

    return err;
}


#if 0
static bool
scanline(tdico *dico, char *s, char *r, bool err)
/* scan host code line s for macro substitution.  r=result line */
{
    int i, k, ls, level, nd, nnest;
    bool spice3;
    char c, d;

    Strbig(Llen, q);
    Strbig(Llen, t);
    Str(20, u);
    spice3 = cpos('3', dico->option) > 0; /* we had -3 on the command line */
    i = 0;
    ls = length(s);
    scopy(r, "");
    err = 0;
    pscopy(u, s, 1, 3);

    if ((ls > 7) && steq(u, "**&")) {
        /* special Comment **&AC #... */
        pscopy(r, s, 1, 7);
        i = 7;
    }

    while ((i < ls) && !err) {
        i++;
        c = s[i - 1];

        if (c == Psp) {

            /* try ps expression syntax */
            k = i;
            nnest = 1;

            do
            {
                k++;
                d = s[k - 1];
                if (d == '{')
                    nnest++;
                else if (d == '}')
                    nnest--;

            } while ((nnest != 0) && (d != '\0'));

            if (d == '\0') {
                err = message(dico, "Closing \"}\" not found.");
            } else {
                pscopy(t, s, i + 1, k - i - 1);
                if (dico->hs_compatibility && (strcasecmp(t, "LAST") == 0)) {
                    strcpy(q, "last");
                    err = 0;
                } else {
                    err = evaluate(dico, q, t, 0);
                }
            }

            i = k;

            if (!err)           /* insert number */
                sadd(r, q);
            else
                err = message(dico, "%s", s);

        } else if (c == Intro) {

            Inc(i);
            while ((i < ls) && (s[i - 1] <= ' '))
                i++;

            k = i;

            if (s[k - 1] == '(') {
                /* sub-formula */
                level = 1;

                do
                {
                    k++;
                    if (k > ls)
                        d = '\0';
                    else
                        d = s[k - 1];

                    if (d == '(')
                        level++;
                    else if (d == ')')
                        level--;

                } while ((k <= ls) && !((d == ')') && (level <= 0)));

                if (k > ls) {
                    err = message(dico, "Closing \")\" not found.");
                } else {
                    pscopy(t, s, i + 1, k - i - 1);
                    err = evaluate(dico, q, t, 0);
                }

                i = k;

            } else {

                /* simple identifier may also be string */
                do
                {
                    k++;
                    if (k > ls)
                        d = '\0';
                    else
                        d = s[k - 1];

                } while ((k <= ls) && (d > ' '));

                pscopy(t, s, i, k - i);
                err = evaluate(dico, q, t, 1);
                i = k - 1;
            }

            if (!err) /* insert the number */
                sadd(r, q);
            else
                message(dico, "%s", s);

        } else if (c == Nodekey) {
            /* follows: a node keyword */

            do
                i++;
            while (s[i - 1] <= ' ');

            k = i;

            do
                k++;
            while ((k <= ls) && alfanum(s[k - 1]));

            pscopy(q, s, i, k - i);
            nd = parsenode(Addr(dico->nodetab), q);

            if (!spice3)
                stri(nd, q);    /* substitute by number */

            sadd(r, q);
            i = k - 1;

        } else {

            if (!spice3)
                c = upcase(c);

            cadd(r, c);         /* c<>Intro */
        }
    }

    return err;
}
#endif


/********* interface functions for spice3f5 extension ***********/

static int
insertnumber(tdico *dico, int i, char *s, SPICE_DSTRINGPTR ustr_p)
/* insert u in string s in place of the next placeholder number */
{
    const char *u = spice_dstring_value(ustr_p);

    char buf[ACT_CHARACTS+1];

    long id = 0;
    int  n  = 0;

    char *p = strstr(s+i, "numparm__");

    if (p &&
        (1 == sscanf(p, "numparm__%8lx%n", &id, &n)) &&
        (n == ACT_CHARACTS) &&
        (id > 0) && (id < dynsubst + 1) &&
        (snprintf(buf, sizeof(buf), "%-17s", u) == ACT_CHARACTS))
    {
        memcpy(p, buf, ACT_CHARACTS);
        return (int)(p - s) + ACT_CHARACTS;
    }

    message
        (dico,
         "insertnumber: fails.\n"
         "  s+i = \"%s\" u=\"%s\" id=%ld",
         s+i, u, id);

    /* swallow everything on failure */
    return i + (int) strlen(s+i);
}


bool
nupa_substitute(tdico *dico, char *s, char *r, bool err)
/* s: pointer to original source line.
   r: pointer to result line, already heavily modified wrt s
   anywhere we find a 10-char numstring in r, substitute it.
   bug: wont flag overflow!
*/
{
    int i, k, ls, level, nnest, ir;
    char c, d;
    SPICE_DSTRING qstr;         /* temp result dynamic string */
    SPICE_DSTRING tstr;         /* temp dynamic string */

    spice_dstring_init(&qstr);
    spice_dstring_init(&tstr);
    i = 0;
    ls = length(s);
    err = 0;
    ir = 0;

    while ((i < ls) && !err) {
        i++;
        c = s[i - 1];

        if (c == Psp) {
            /* try ps expression syntax */
            k = i;
            nnest = 1;

            do
            {
                k++;
                d = s[k - 1];
                if (d == '{')
                    nnest++;
                else if (d == '}')
                    nnest--;

            } while ((nnest != 0) && (d != '\0'));

            if (d == '\0') {
                err = message(dico, "Closing \"}\" not found.");
            } else {
                pscopy(&tstr, s, i , k - i - 1);
                /* exeption made for .meas */
                if (strcasecmp(spice_dstring_value(&tstr), "LAST") == 0) {
                    spice_dstring_reinit(&qstr);
                    sadd(&qstr, "last");
                    err = 0;
                } else {
                    err = evaluate(dico, &qstr, spice_dstring_value(&tstr), 0);
                }
            }

            i = k;
            if (!err)
                ir = insertnumber(dico, ir, r, &qstr);
            else
                err = message(dico, "Cannot compute substitute");

        } else if (c == Intro) {
            /* skip "&&" which may occur in B source */

            if ((i + 1 < ls) && (s[i] == Intro)) {
                i++;
                continue;
            }

            i++;
            while ((i < ls) && (s[i - 1] <= ' '))
                i++;

            k = i;

            if (s[k - 1] == '(') {
                /* sub-formula */
                level = 1;

                do
                {
                    k++;
                    if (k > ls)
                        d = '\0';
                    else
                        d = s[k - 1];

                    if (d == '(')
                        level++;
                    else if (d == ')')
                        level--;

                } while ((k <= ls) && !((d == ')') && (level <= 0)));

                if (k > ls) {
                    err = message(dico, "Closing \")\" not found.");
                } else {
                    pscopy(&tstr, s, i, k - i - 1);
                    err = evaluate(dico, &qstr, spice_dstring_value(&tstr), 0);
                }

                i = k;

            } else {
                /* simple identifier may also be string? */

                do
                {
                    k++;
                    if (k > ls)
                        d = '\0';
                    else
                        d = s[k - 1];

                } while ((k <= ls) && (d > ' '));

                pscopy(&tstr, s, i-1, k - i);
                err = evaluate(dico, &qstr, spice_dstring_value(&tstr), 1);
                i = k - 1;
            }

            if (!err)
                ir = insertnumber(dico, ir, r, &qstr);
            else
                message(dico, "Cannot compute &(expression)");
        }
    }

    spice_dstring_free(&qstr);
    spice_dstring_free(&tstr);

    return err;
}


static unsigned char
getword(char *s, SPICE_DSTRINGPTR tstr_p, int after, int *pi)
/* isolate a word from s after position "after". return i= last read+1 */
{
    int i = *pi;
    int ls;
    unsigned char key;
    char *t_p;

    i = after;
    ls = length(s);

    do
        i++;
    while ((i < ls) && !alfa(s[i - 1]));

    spice_dstring_reinit(tstr_p);

    while ((i <= ls) && (alfa(s[i - 1]) || num(s[i - 1]))) {
        cadd(tstr_p, upcase(s[i - 1]));
        i++;
    }

    t_p = spice_dstring_value(tstr_p);

    if (t_p[0])
        key = keyword(&keyS, tstr_p);
    else
        key = 0;

    *pi = i;

    return key;
}


static char
getexpress(char *s, SPICE_DSTRINGPTR tstr_p, int *pi)
/* returns expression-like string until next separator
   Input  i=position before expr, output  i=just after expr, on separator.
   returns tpe=='R' if (numeric, 'S' if (string only
*/
{
    int i = *pi;
    int ia, ls, level;
    char c, d, tpe;
    bool comment = 0;

    ls = length(s);
    ia = i + 1;

    while ((ia < ls) && (s[ia - 1] <= ' '))
        ia++;                   /*white space ? */

    if (s[ia - 1] == '"') {
        /* string constant */
        ia++;
        i = ia;

        while ((i < ls) && (s[i - 1] != '"'))
            i++;

        tpe = 'S';

        do
            i++;
        while ((i <= ls) && (s[i - 1] <= ' '));

    } else {

        if (s[ia - 1] == '{')
            ia++;

        i = ia - 1;

        do
        {
            i++;

            if (i > ls)
                c = ';';
            else
                c = s[i - 1];

            if (c == '(') {
                /* sub-formula */
                level = 1;
                do
                {
                    i++;

                    if (i > ls)
                        d = '\0';
                    else
                        d = s[i - 1];

                    if (d == '(')
                        level++;
                    else if (d == ')')
                        level--;

                } while ((i <= ls) && !((d == ')') && (level <= 0)));
            }

            /* buggy? */
            if ((c == '/') || (c == '-'))
                comment = (s[i] == c);

        } while (!((cpos (c, ",;)}") >= 0) || comment)); /* legal separators */

        tpe = 'R';
    }

    pscopy(tstr_p, s, ia-1, i - ia);

    if (s[i - 1] == '}')
        i++;

    if (tpe == 'S')
        i++;                    /* beyond quote */

    *pi = i;

    return tpe;
}


bool
nupa_assignment(tdico *dico, char *s, char mode)
/* is called for all 'Param' lines of the input file.
   is also called for the params: section of a subckt .
   mode='N' define new local variable, else global...
   bug: we cannot rely on the transformed line, must re-parse everything!
*/
{
    /* s has the format: ident = expression; ident= expression ...  */
    int i, j, ls;
    unsigned char key;
    bool error, err;
    char dtype;
    int wval = 0;
    double rval = 0.0;
    char *t_p;                  /* dstring contents value */
    SPICE_DSTRING tstr;         /* temporary dstring */
    SPICE_DSTRING ustr;         /* temporary dstring */

    spice_dstring_init(&tstr);
    spice_dstring_init(&ustr);
    ls = length(s);
    error = 0;
    i = 0;
    j = spos_("//", s);                /* stop before comment if any */

    if (j >= 0)
        ls = j;

    /* bug: doesnt work. need to  revise getexpress ... !!! */
    i = 0;

    while ((i < ls) && (s[i] <= ' '))
        i++;

    if (s[i] == Intro)
        i++;

    if (s[i] == '.')            /* skip any dot keyword */
        while (s[i] > ' ')
            i++;

    while ((i < ls) && !error) {

        key = getword(s, &tstr, i, &i);
        t_p = spice_dstring_value(&tstr);
        if ((t_p[0] == '\0') || (key > 0))
            error = message(dico, " Identifier expected");

        if (!error) {
            /* assignment expressions */
            while ((i <= ls) && (s[i - 1] != '='))
                i++;

            if (i > ls)
                error = message(dico, " = sign expected .");

            dtype = getexpress(s, &ustr, &i);

            if (dtype == 'R') {
                rval = formula(dico, spice_dstring_value(&ustr), &error);
                if (error) {
                    message(dico, " Formula() error.");
                    fprintf(stderr, "      %s\n", s);
                }
            } else if (dtype == 'S') {
                wval = i;
            }

            err = define(dico, spice_dstring_value(&tstr), mode /* was ' ' */ ,
                         dtype, rval, wval, NULL, NULL);
            error = error || err;
        }

        if ((i < ls) && (s[i - 1] != ';'))
            error = message(dico, " ; sign expected.");
        /* else
           i++; */
    }

    spice_dstring_free(&tstr);
    spice_dstring_free(&ustr);

    return error;
}


bool
nupa_subcktcall(tdico *dico, char *s, char *x, bool err)
/* s= a subckt define line, with formal params.
   x= a matching subckt call line, with actual params
*/
{
    int n, i, j, k, g, h, narg = 0, ls, nest;
    SPICE_DSTRING subname;
    SPICE_DSTRING tstr;
    SPICE_DSTRING ustr;
    SPICE_DSTRING vstr;
    SPICE_DSTRING idlist;
    SPICE_DSTRING parsebuf;
    char *buf, *token;
    char *t_p;
    char *u_p;
    bool found;

    spice_dstring_init(&subname);
    spice_dstring_init(&tstr);
    spice_dstring_init(&ustr);
    spice_dstring_init(&vstr);
    spice_dstring_init(&idlist);

    /*
      skip over instance name -- fixes bug where instance 'x1' is
      same name as subckt 'x1'
    */
    while (*x != ' ')
        x++;

    /***** first, analyze the subckt definition line */
    n = 0;                      /* number of parameters if any */
    ls = length(s);
    j = spos_("//", s);

    if (j >= 0)
        pscopy_up(&tstr, s, 0, j);
    else
        scopy_up(&tstr, s);

    j = spos_("SUBCKT", spice_dstring_value(&tstr));

    if (j >= 0) {
        j = j + 6;              /* fetch its name - skip subckt */
        t_p = spice_dstring_value(&tstr);
        while ((j < ls) && (t_p[j] <= ' '))
            j++;

        while (t_p[j] != ' ') {
            cadd(&subname, t_p[j]);
            j++;
        }
    } else {
        err = message(dico, " ! a subckt line!");
    }

    i = spos_("PARAMS:", spice_dstring_value(&tstr));

    if (i >= 0) {
        const char *optr, *jptr;

        pscopy(&tstr, spice_dstring_value(&tstr), i + 7, spice_dstring_length(&tstr));

        /* search identifier to the left of '=' assignments */

        for (optr = spice_dstring_value(&tstr);
             (jptr = strchr(optr, '=')) != NULL;
             optr = jptr + 1)
        {
            const char *kptr, *hptr;

            /* skip "==" */
            if (jptr[1] == '=') {
                jptr++;
                continue;
            }

            /* skip "<=" ">=" "!=" */
            if (jptr > optr && strchr("<>!", jptr[-1]))
                continue;

            kptr = jptr;
            while (--kptr >= optr && isspace(*kptr))
                ;

            hptr = kptr;
            while (hptr >= optr && alfanum(*hptr))
                hptr--;

            if (hptr < kptr && alfa(hptr[1])) {
                while (hptr++ < kptr)
                    cadd(&idlist, *hptr);

                sadd(&idlist, "=$;");
                n++;
            } else {
                message(dico, "identifier expected.");
            }
        }
    }

    /***** next, analyze the circuit call line */
    if (!err) {

        narg = 0;
        j = spos_("//", x);

        if (j >= 0) {
            pscopy_up(&tstr, x, 0, j);
        } else {
            scopy_up(&tstr, x);
            j = 0;
        }

        ls = spice_dstring_length(&tstr);

        spice_dstring_init(&parsebuf);
        scopyd(&parsebuf, &tstr);
        buf = spice_dstring_value(&parsebuf);

        found = 0;
        token = strtok(buf, " "); /* a bit more exact - but not sufficient everytime */
        j = j + (int) strlen(token) + 1;
        if (strcmp(token, spice_dstring_value(&subname)))
            while ((token = strtok(NULL, " ")) != NULL) {
                if (!strcmp(token, spice_dstring_value(&subname))) {
                    found = 1;
                    break;
                }
                j = j + (int) strlen(token) + 1;
            }

        spice_dstring_free(&parsebuf);

        /*  make sure that subname followed by space */
        if (found) {
            j = j + spice_dstring_length(&subname) + 1; /* 1st position of arglist: j */

            t_p = spice_dstring_value(&tstr);
            while ((j < ls) && ((t_p[j] <= ' ') || (t_p[j] == ',')))
                j++;

            while (j < ls) {

                /* try to fetch valid arguments */
                k = j;
                spice_dstring_reinit(&ustr);

                if (t_p[k] == Intro) {

                    /* handle historical syntax... */
                    if (alfa(t_p[k + 1])) {
                        k++;
                    } else if (t_p[k + 1] == '(') {
                        /* transform to braces... */
                        k++;
                        t_p[k] = '{';
                        g = k;
                        nest = 1;

                        while ((nest > 0) && (g < ls)) {
                            g++;
                            if (t_p[g] == '(')
                                nest++;
                            else if (t_p[g] == ')')
                                nest--;
                        }

                        if ((g < ls) && (nest == 0))
                            t_p[g] = '}';
                    }
                }

                if (alfanum(t_p[k]) || t_p[k] == '.') {
                    /* number, identifier */
                    h = k;
                    while (t_p[k] > ' ')
                        k++;
                    pscopy(&ustr, spice_dstring_value(&tstr), h, k - h);
                    j = k;
                } else if (t_p[k] == '{') {
                    getexpress(spice_dstring_value(&tstr), &ustr, &j);
                    j--;       /* confusion: j was in Turbo Pascal convention */
                } else {
                    j++;
                    if (t_p[k] > ' ') {
                        spice_dstring_append(&vstr, "Subckt call, symbol ", -1);
                        cadd(&vstr, t_p[k]);
                        sadd(&vstr, " not understood");
                        message(dico, "%s", spice_dstring_value(&vstr));
                    }
                }

                u_p = spice_dstring_value(&ustr);
                if (u_p[0]) {
                    narg++;
                    k = cpos('$', spice_dstring_value(&idlist));
                    if (k >= 0) {
                        /* replace dollar with expression string u */
                        pscopy(&vstr, spice_dstring_value(&idlist), 0, k);
                        sadd(&vstr, spice_dstring_value(&ustr));
                        pscopy(&ustr, spice_dstring_value(&idlist), k+1, spice_dstring_length(&idlist));
                        scopyd(&idlist, &vstr);
                        sadd(&idlist, spice_dstring_value(&ustr));
                    }
                }
            }
        } else {
            message(dico, "Cannot find called subcircuit");
        }
    }

    /***** finally, execute the multi-assignment line */
    dicostack(dico, Push);      /* create local symbol scope */

    if (narg != n) {
        err = message(dico,
                      " Mismatch: %d  formal but %d actual params.\n"
                      "%s",
                      n, narg, spice_dstring_value(&idlist));
        /* ;} else { debugwarn(dico, idlist) */
    }

    err = nupa_assignment(dico, spice_dstring_value(&idlist), 'N');

    spice_dstring_free(&subname);
    spice_dstring_free(&tstr);
    spice_dstring_free(&ustr);
    spice_dstring_free(&vstr);
    spice_dstring_free(&idlist);

    return err;
}


void
nupa_subcktexit(tdico *dico)
{
    dicostack(dico, Pop);
}
