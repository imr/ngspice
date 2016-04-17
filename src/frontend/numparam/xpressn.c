/* xpressn.c                Copyright (C)  2002    Georg Post

   This file is part of Numparam, see:  readme.txt
   Free software under the terms of the GNU Lesser General Public License
*/

#include "ngspice/ngspice.h"

#include "general.h"
#include "numparam.h"
#include "ngspice/cpdefs.h"
#include "ngspice/ftedefs.h"
#include "ngspice/dvec.h"
#include "../frontend/variable.h"
#include "ngspice/compatmode.h"


/* random numbers in /maths/misc/randnumb.c */
#include "ngspice/randnumb.h"

/************ keywords ************/

extern char *nupa_inst_name;    /* see spicenum.c */
extern long dynsubst;           /* see inpcom.c */

#define ACT_CHARACTS 25      /* actual string length to be inserted and replaced */

#define  S_init   0
#define  S_atom   1
#define  S_binop  2
#define  S_unop   3
#define  S_stop   4


static double
ternary_fcn(double conditional, double if_value, double else_value)
{
    if (conditional != 0.0)
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


static const char *fmathS =     /* all math functions */
    "SQR SQRT SIN COS EXP LN ARCTAN ABS POW PWR MAX MIN INT LOG LOG10 SINH COSH"
    " TANH TERNARY_FCN AGAUSS SGN GAUSS UNIF AUNIF LIMIT CEIL FLOOR"
    " ASIN ACOS ATAN ASINH ACOSH ATANH TAN NINT";


enum {
    XFU_SQR = 1, XFU_SQRT, XFU_SIN, XFU_COS, XFU_EXP, XFU_LN, XFU_ARCTAN, XFU_ABS, XFU_POW, XFU_PWR, XFU_MAX, XFU_MIN, XFU_INT, XFU_LOG, XFU_LOG10, XFU_SINH, XFU_COSH,
    XFU_TANH, XFU_TERNARY_FCN, XFU_AGAUSS, XFU_SGN, XFU_GAUSS, XFU_UNIF, XFU_AUNIF, XFU_LIMIT, XFU_CEIL, XFU_FLOOR,
    XFU_ASIN, XFU_ACOS, XFU_ATAN, XFU_ASINH, XFU_ACOSH, XFU_ATANH, XFU_TAN, XFU_NINT
};


static double
mathfunction(int f, double z, double x)
/* the list of built-in functions. Patch 'fmath', here and near line 888 to get more ...*/
{
    double y;
    switch (f)
    {
    case XFU_SQR:
        y = x * x;
        break;
    case XFU_SQRT:
        y = sqrt(x);
        break;
    case XFU_SIN:
        y = sin(x);
        break;
    case XFU_COS:
        y = cos(x);
        break;
    case XFU_EXP:
        y = exp(x);
        break;
    case XFU_LN:
        y = log(x);
        break;
    case XFU_ARCTAN:
        y = atan(x);
        break;
    case XFU_ABS:
        y = fabs(x);
        break;
    case XFU_POW:
        y = pow(z, x);
        break;
    case XFU_PWR:
        y = pow(fabs(z), x);
        break;
    case XFU_MAX:
        y = MAX(x, z);
        break;
    case XFU_MIN:
        y = MIN(x, z);
        break;
    case XFU_INT:
        y = trunc(x);
        break;
    case XFU_NINT:
        /* round to "nearest integer",
         *   round half-integers to the nearest even integer
         *   rely on default rounding mode of IEEE 754 to do so
         */
        y = nearbyint(x);
        break;
    case XFU_LOG:
        y = log(x);
        break;
    case XFU_LOG10:
        y = log10(x);
        break;
    case XFU_SINH:
        y = sinh(x);
        break;
    case XFU_COSH:
        y = cosh(x);
        break;
    case XFU_TANH:
        y = tanh(x);
        break;
    case XFU_SGN:
        if (x > 0)
            y = 1.;
        else if (x == 0)
            y = 0.;
        else
            y = -1.;
        break;
    case XFU_CEIL:
        y = ceil(x);
        break;
    case XFU_FLOOR:
        y = floor(x);
        break;
    case XFU_ASIN:
        y = asin(x);
        break;
    case XFU_ACOS:
        y = acos(x);
        break;
    case XFU_ATAN:
        y = atan(x);
        break;
    case XFU_ASINH:
        y = asinh(x);
        break;
    case XFU_ACOSH:
        y = acosh(x);
        break;
    case XFU_ATANH:
        y = atanh(x);
        break;
    case XFU_TAN:
        y = tan(x);
        break;
    default:
        y = x;
        break;
    }

    return y;
}


#ifdef __GNUC__
static bool message(dico_t *dico, const char *fmt, ...)
    __attribute__ ((format (__printf__, 2, 3)));
#endif


static bool
message(dico_t *dico, const char *fmt, ...)
{
    va_list ap;

    char *srcfile = spice_dstring_value(&(dico->srcfile));

    if (srcfile && *srcfile)
        fprintf(stderr, "%s:", srcfile);

    if (dico->srcline >= 0)
        fprintf
            (stderr,
             "Original line no.: %d, new internal line no.: %d:\n",
             dico->oldline, dico->srcline);

    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);

    dico->errcount++;

    return 1; /* error! */
}


/************ the input text symbol table (dictionary) *************/

void
initdico(dico_t *dico)
{
    int asize = 10;           /* default allocation depth of the synbol stack */
    COMPATMODE_T compat_mode;

    spice_dstring_init(&(dico->option));
    spice_dstring_init(&(dico->srcfile));

    dico->srcline = -1;
    dico->errcount = 0;

    spice_dstring_init(&(dico->lookup_buf));

    dico->symbols = TMALLOC(NGHASHPTR, asize);
    dico->inst_name = TMALLOC(char*, asize);
    dico->max_stack_depth = asize;
    dico->stack_depth = 0;              /* top of the stack */

    dico->symbols[0] = nghash_init(NGHASH_MIN_SIZE);
    nghash_unique(dico->symbols[0], TRUE); /* no rewrite of global symbols */

    dico->inst_symbols = NULL;          /* instance qualified are lazily allocated */

    compat_mode = ngspice_compat_mode();

    if (compat_mode == COMPATMODE_HS)
        dico->hs_compatibility = 1;
    else
        dico->hs_compatibility = 0;
}


void
dico_free_entry(entry_t *entry)
{
    if (entry->symbol)
        txfree(entry->symbol);

    txfree(entry);
}


/* local semantics for parameters inside a subckt */
/* arguments as wll as .param expressions  */
/* to do:  scope semantics ?
   "params:" and all new symbols should have local scope inside subcircuits.
   redefinition of old symbols gives a warning message.
*/

static void
dicostack_push(dico_t *dico)
/* push operation for nested subcircuit locals */
{
    dico->stack_depth++;

    if (dico->stack_depth >= dico->max_stack_depth) {
        int asize = (dico->max_stack_depth *= 2);
        dico->symbols = TREALLOC(NGHASHPTR, dico->symbols, asize);
        dico->inst_name = TREALLOC(char*, dico->inst_name, asize);
    }

    /* lazy allocation - don't allocate space if we can help it */
    dico->symbols[dico->stack_depth] = NULL;
    dico->inst_name[dico->stack_depth] = nupa_inst_name;
}


static void
dicostack_pop(dico_t *dico)
/* pop operation for nested subcircuit locals */
{
    char *inst_name;            /* name of subcircuit instance */
    char *param_p;              /* qualified inst parameter name */
    entry_t *entry;             /* current entry */
    NGHASHPTR htable_p;         /* current hash table */
    NGHASHITER iter;            /* hash iterator - thread safe */

    if (dico->stack_depth <= 0) {
        message(dico, " Subckt Stack underflow.\n");
        return;
    }

    /* -----------------------------------------------------------------
     * Keep instance parameters around by transferring current local
     * scope variables to an instance qualified hash table.
     * ----------------------------------------------------------------- */
    inst_name = dico->inst_name[dico->stack_depth];
    htable_p = dico->symbols[dico->stack_depth];
    if (htable_p) {
        SPICE_DSTRING param_name; /* build a qualified name */
        spice_dstring_init(&param_name);

        NGHASH_FIRST(&iter);
        for (entry = (entry_t *) nghash_enumerateRE(htable_p, &iter);
             entry;
             entry = (entry_t *) nghash_enumerateRE(htable_p, &iter))
        {
            spice_dstring_reinit(&param_name);
            param_p = spice_dstring_print(&param_name, "%s.%s",
                                          inst_name, entry->symbol);
            nupa_add_inst_param(param_p, entry->vl);
            dico_free_entry(entry);
        }
        nghash_free(htable_p, NULL, NULL);
        spice_dstring_free(&param_name);
    }
    tfree(inst_name);

    dico->inst_name[dico->stack_depth] = NULL;
    dico->symbols[dico->stack_depth] = NULL;
    dico->stack_depth--;
}


int
donedico(dico_t *dico)
{
    int sze = nghash_get_size(dico->symbols[0]);
    return sze;
}


/* -----------------------------------------------------------------
 * Now entryb works on the given hash table hierarchy.   First
 * look thru the stack of local symbols and then look at the global
 * symbols in that order.
 * ----------------------------------------------------------------- */
static entry_t *
entrynb(dico_t *dico, char *s)
{
    int depth;                  /* stack depth */
    entry_t *entry;             /* search hash table */
    NGHASHPTR htable_p;         /* hash table */

    /* look at the current scope and then backup the stack */
    for (depth = dico->stack_depth; depth >= 0; depth--) {
        htable_p = dico->symbols[depth];
        if (htable_p) {
            entry = (entry_t *) nghash_find(htable_p, s);
            if (entry)
                return (entry);
        }
    }

    return NULL;
}


char
getidtype(dico_t *dico, char *s)
/* test if identifier s is known. Answer its type, or '?' if not in table */
{
    entry_t *entry = entrynb(dico, s);

    if (entry)
        return entry->tp;

    return '?';
}


static double
fetchnumentry(dico_t *dico, char *s, bool *perr)
{
    entry_t *entry = entrynb(dico, s);

    while (entry && (entry->tp == 'P'))
        entry = entry->pointer;

    if (entry && (entry->tp == 'R'))
        return entry->vl;

    *perr = message(dico, "Undefined number [%s]\n", s);
    return 0.0;
}


/*******  writing dictionary entries *********/

entry_t *
attrib(dico_t *dico, NGHASHPTR htable_p, char *t, char op)
{
    /* seek or attribute dico entry number for string t.
       Option  op='N' : force a new entry, if tos>level and old is  valid.
    */
    entry_t *entry;             /* symbol table entry */

    entry = (entry_t *) nghash_find(htable_p, t);
    if (entry && (op == 'N') &&
        (entry->level < dico->stack_depth) && (entry->tp != '?'))
    {
        entry = NULL;
    }

    if (!entry) {
        entry = TMALLOC(entry_t, 1);
        entry->symbol = strdup(t);
        entry->tp = '?';      /* signal Unknown */
        entry->level = dico->stack_depth;
        nghash_insert(htable_p, t, entry);
    }

    return entry;
}


/* user defined delete function:
 *   free the dictionary entries malloc'ed above
 * will be called by nghash_free() in nupa_del_dicoS()
 */

void
del_attrib(void *entry_p)
{
    entry_t *entry = (entry_t*) entry_p;
    if(entry) {
        tfree(entry->symbol);
        tfree(entry);
    }
}


static bool
nupa_define(dico_t *dico,
       char *t,                 /* identifier to define */
       char op,                 /* option */
       char tpe,                /* type marker */
       double z,                /* float value if any */
       int w,                   /* integer value if any */
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
    entry_t *entry;             /* spice table entry */
    NGHASHPTR htable_p;         /* hash table */

    /* can't be lazy anymore */
    if (!(dico->symbols[dico->stack_depth]))
        dico->symbols[dico->stack_depth] = nghash_init(NGHASH_MIN_SIZE);

    htable_p = dico->symbols[dico->stack_depth];

    entry = attrib(dico, htable_p, t, op);
    err = 0;

    if (!entry) {

        err = message(dico, " Symbol table overflow\n");

    } else {

        if (entry->tp == 'P')
            entry = entry->pointer; /* pointer indirection */

        if (entry)
            c = entry->tp;
        else
            c = ' ';

        if ((c == 'R') || (c == 'S') || (c == '?')) {

            entry->vl = z;
            entry->tp = tpe;
            entry->ivl = w;
            entry->sbbase = base;
            /* if ((c != '?') && (i <= dico->stack[dico->tos])) { */
            if (c == '?')
                entry->level = dico->stack_depth; /* promote! */

            /* warn about re-write to a global scope! */
            if (entry->level < dico->stack_depth)
                warn = message(dico, "%s:%d overwritten.\n", t, entry->level);

        } else {
            /* error message for redefinition of symbols */
            message(dico, "%s is already used,\n cannot be redefined\n", t);
        }
    }

    return err;
}


bool
defsubckt(dico_t *dico, char *s, int w, char categ)
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
        err = nupa_define(dico, spice_dstring_value(&ustr), ' ', categ, 0.0, w, NULL);
        spice_dstring_free(&ustr);
    } else {
        err = message(dico, "Subcircuit or Model without name.\n");
    }

    return err;
}


int
findsubckt(dico_t *dico, char *s, SPICE_DSTRINGPTR subname)
/* input: s is a subcircuit invocation line.
   returns 0 if not found, else the stored definition line number value
   and the name in string subname  */
{
    entry_t *entry;             /* symbol table entry */
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
    entry = entrynb(dico, spice_dstring_value(&ustr));

    if (entry && (entry->tp == 'U')) {
        line = entry->ivl;
        scopyd(subname, &ustr);
    } else {
        line = 0;
        spice_dstring_reinit(subname);
        message(dico, "Cannot find subcircuit.\n");
    }

    return line;
}


/************ input scanner stuff **************/

static unsigned char
keyword(const char *keys, const char *s, const char *s_end)
{
    /* return 0 if s not found in list keys, else the ordinal number */
    unsigned char j = 1;

    if (!*s)
        return 0;

    for (;;) {
        const char *p = s;
        while ((p < s_end) && (upcase(*p) == *keys))
            p++, keys++;
        if ((p >= s_end) && (*keys <= ' '))
            return j;
        keys = strchr(keys, ' ');
        if (!keys)
            return 0;
        keys++;
        j++;
    }
}


static double
parseunit(const char *s)
/* the Spice suffixes */
{
    switch (toupper_c(s[0]))
    {
    case 'T':  return 1e12;
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


static const char *
fetchid(const char *s, const char *s_end)
{
    for (; s < s_end; s++)
        if (!(alfanum(*s) || *s == '.'))
            return s;

    return s;
}


static double
fetchnumber(dico_t *dico, const char **pi, bool *perror)
/* parse a Spice number in string s */
{
    double u;
    int n = 0;

    const char *s = *pi;

    if (1 != sscanf(s, "%lG%n", &u, &n)) {

        *perror = message(dico, "Number format error: \"%s\"\n", s);

        return 0.0;             /* FIXME return NaN */

    }

    u *= parseunit(s + n);

    /* swallow unit
     *   FIXME `100MegBaz42' should emit an error message
     *   FIXME should we allow whitespace ?   `100 MEG' ?
     */

    while (s[n] && alfa(s[n]))
        n++;

    *pi += n;

    return u;
}


static char
fetchoperator(dico_t *dico,
              const char *s_end,
              const char **pi,
              unsigned char *pstate, unsigned char *plevel,
              bool *perror)
/* grab an operator from string s and advance scan index pi.
   each operator has: one-char alias, precedence level, new interpreter state.
*/
{
    const char *iptr = *pi;
    unsigned char state = *pstate;
    unsigned char level = *plevel;
    bool error = *perror;
    char c, d;

    c = *iptr++;

    d = *iptr;
    if (iptr >= s_end)
        d = '\0';

    if ((c == '!') && (d == '=')) {
        c = '#';
        iptr++;
    } else if ((c == '<') && (d == '>')) {
        c = '#';
        iptr++;
    } else if ((c == '<') && (d == '=')) {
        c = 'L';
        iptr++;
    } else if ((c == '>') && (d == '=')) {
        c = 'G';
        iptr++;
    } else if ((c == '*') && (d == '*')) {
        c = '^';
        iptr++;
    } else if ((c == '=') && (d == '=')) {
        iptr++;
    } else if ((c == '&') && (d == '&')) {
        c = 'A';
        iptr++;
    } else if ((c == '|') && (d == '|')) {
        c = 'O';
        iptr++;
    }

    if ((c == '+') || (c == '-')) {
        state = S_binop;        /* pending operator */
        level = 4;
    } else if ((c == '*') || (c == '/') || (c == '%') || (c == '\\')) {
        state = S_binop;
        level = 3;
    } else if (c == '^') {
        state = S_binop;
        level = 2;
    } else if (cpos(c, "=<>#GL") >= 0) {
        state = S_binop;
        level = 5;
    } else if (c == 'A') {
        state = S_binop;
        level = 6;
    } else if (c == 'O') {
        state = S_binop;
        level = 7;
    } else if (c == '!') {
        state = S_unop;
    } else if (c == '?') {
        state = S_binop;
        level = 9;
    } else if (c == ':') {
        state = S_binop;
        level = 8;
    } else {
        state = S_init;
        if (c > ' ')
            error = message(dico, "Syntax error: letter [%c]\n", c);
    }

    *pi = iptr;
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
        x = pow(fabs(x), y);
        break;
    case 'A':                   /* && */
        x = ((x != 0.0) && (y != 0.0)) ? 1.0 : 0.0;
        break;
    case 'O':                   /* || */
        x = ((x != 0.0) || (y != 0.0)) ? 1.0 : 0.0;
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
        x = np_trunc(fabs(x / y));
        break;
    }

    return x;
}


#define nprece 9 /* maximal nb of precedence levels */

static double
formula(dico_t *dico, const char *s, const char *s_end, bool *perror)
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
    bool error = *perror;
    bool negate = 0;
    unsigned char state, oldstate, topop, ustack, level, fu;
    double u = 0.0;
    double accu[nprece + 1];
    char oper[nprece + 1];
    char uop[nprece + 1];
    int i, natom;
    bool ok;
    SPICE_DSTRING tstr;
    const char *s_orig = s;

    spice_dstring_init(&tstr);

    for (i = 0; i <= nprece; i++) {
        accu[i] = 0.0;
        oper[i] = ' ';
    }

    /* trim trailing whitespace */
    while ((s_end > s) && (s_end[-1] <= ' '))
        s_end--;

    state = S_init;
    natom = 0;
    ustack = 0;
    topop = 0;
    oldstate = S_init;
    fu = 0;
    error = 0;
    level = 0;

    while ((s < s_end) && !error) {
        char c = *s;
        if (c == '(') {
            /* sub-formula or math function */
            double v = 1.0, w = 0.0;
            /* new: must support multi-arg functions */
            const char *kptr = ++s;
            const char *arg2 = NULL;
            const char *arg3 = NULL;
            char d;

            level = 1;
            do
            {
                d = *kptr++;
                if (kptr > s_end)
                    d = '\0';

                if (d == '(')
                    level++;
                else if (d == ')')
                    level--;

                if ((d == ',') && (level == 1)) {
                    if (arg2 == NULL)
                        arg2 = kptr;
                    else
                        arg3 = kptr; /* kludge for more than 2 args (ternary expression) */
                }                 /* comma list? */

            } while ((kptr <= s_end) && !((d == ')') && (level <= 0)));

            // fixme, here level = 0 !!!!! (almost)

            if (kptr > s_end) {
                error = message(dico, "Closing \")\" not found.\n");
                natom++;        /* shut up other error message */
            } else {
                if (arg2 > s) {
                    v = formula(dico, s, arg2 - 1, &error);
                    s = arg2;
                }
                if (arg3 > s) {
                    w = formula(dico, s, arg3 - 1, &error);
                    s = arg3;
                }
                u = formula(dico, s, kptr - 1, &error);
                state = S_atom;
                if (fu > 0) {
                    if ((fu == XFU_TERNARY_FCN))
                        u = ternary_fcn(v, w, u);
                    else if ((fu == XFU_AGAUSS))
                        u = agauss(v, w, u);
                    else if ((fu == XFU_GAUSS))
                        u = gauss(v, w, u);
                    else if ((fu == XFU_UNIF))
                        u = unif(v, u);
                    else if ((fu == XFU_AUNIF))
                        u = aunif(v, u);
                    else if ((fu == XFU_LIMIT))
                        u = limit(v, u);
                    else
                        u = mathfunction(fu, v, u);
                }
            }
            s = kptr;
            fu = 0;
        } else if (alfa(c)) {
            const char *s_next = fetchid(s, s_end);
            fu = keyword(fmathS, s, s_next); /* numeric function? */
            if (fu > 0) {
                state = S_init;  /* S_init means: ignore for the moment */
            } else {
                spice_dstring_reinit(&tstr);
                while (s < s_next)
                    cadd(&tstr, upcase(*s++));
                u = fetchnumentry(dico, spice_dstring_value(&tstr), &error);
                state = S_atom;
            }
            s = s_next;
        } else if (((c == '.') || ((c >= '0') && (c <= '9')))) {
            u = fetchnumber(dico, &s, &error);
            if (negate) {
                u = -1 * u;
                negate = 0;
            }
            state = S_atom;
        } else {
            c = fetchoperator(dico, s_end, &s, &state, &level, &error);
        }

        /* may change c to some other operator char! */
        /* control chars <' '  ignored */

        ok = (oldstate == S_init) || (state == S_init) ||
            ((oldstate == S_atom) && (state == S_binop)) ||
            ((oldstate != S_atom) && (state != S_binop));

        if (oldstate == S_binop && state == S_binop && c == '-') {
            ok = 1;
            negate = 1;
            continue;
        }

        if (!ok)
            error = message(dico, " Misplaced operator\n");

        if (state == S_unop) {
            /* push unary operator */
            uop[++ustack] = c;
        } else if (state == S_atom) {
            /* atom pending */
            natom++;
            if (s >= s_end) {
                state = S_stop;
                level = topop;
            } /* close all ops below */

            while (ustack > 0)
                u = operate(uop[ustack--], u, u);

            accu[0] = u;        /* done: all pending unary operators */
        }

        if ((state == S_binop) || (state == S_stop)) {
            /* do pending binaries of priority Upto "level" */
            for (i = 1; i <= level; i++) {
                if (i < level && oper[i] == ':' && (oper[i+1] == '?' || oper[i+1] == 'x')) {
                    if (oper[i+1] == 'x') {
                        /* this is a `first-of-triple' op */
                        accu[i+1] = accu[i+1];
                        c = 'x';  /* transform next '?' to 'first-of-triple' */
                    } else if (accu[i+1] != 0.0) {
                        /* this is a `true' ternary */
                        accu[i+1] = accu[i];
                        c = 'x';  /* transform next '?' to `first-of-triple' */
                    } else {
                        /* this is a `false' ternary */
                        accu[i+1] = accu[i-1];
                    }
                    accu[i-1] = 0.0;
                    oper[i] = ' ';  /* reset intermediates */
                    i++;
                    accu[i-1] = 0.0;
                    oper[i] = ' ';  /* reset intermediates */
                } else {
                    /* not yet speed optimized! */
                    accu[i] = operate(oper[i], accu[i], accu[i-1]);
                    accu[i-1] = 0.0;
                    oper[i] = ' ';  /* reset intermediates */
                }
            }
            oper[level] = c;

            if (topop < level)
                topop = level;
        }

        if (state != S_init)
            oldstate = state;
    }

    if ((natom == 0) || (oldstate != S_stop))
        error = message(dico, " Expression err: %s\n", s_orig);

    if (negate == 1)
        error = message(dico,
                        " Problem with formula eval -- wrongly determined negation!\n");

    *perror = error;

    spice_dstring_free(&tstr);

    if (error)
        return 1.0;
    else
        return accu[topop];
}


static bool
evaluate(dico_t *dico, SPICE_DSTRINGPTR qstr_p, char *t, unsigned char mode)
{
    /* transform t to result q. mode 0: expression, mode 1: simple variable */
    double u = 0.0;
    int j, lq;
    char dt;
    entry_t *entry;
    bool numeric, done, nolookup;
    bool err;

    spice_dstring_reinit(qstr_p);
    numeric = 0;
    err = 0;

    if (mode == 1) {
        /* string? */
        stupcase(t);
        entry = entrynb(dico, t);
        nolookup = !entry;

        while (entry && (entry->tp == 'P'))
            entry = entry->pointer; /* follow pointer chain */

        /* pointer chain */
        if (entry)
            dt = entry->tp;
        else
            dt = ' ';

        /* data type: Real or String */
        if (dt == 'R') {
            u = entry->vl;
            numeric = 1;
        } else if (dt == 'S') {
            /* suppose source text "..." at */
            j = entry->ivl;
            lq = 0;

            do
            {
                j++;
                lq++;
                dt = /* ibf->bf[j]; */ entry->sbbase[j];

                if (cpos('3', spice_dstring_value(&dico->option)) <= 0)
                    dt = upcase(dt); /* spice-2 */

                done = (dt == '\"') || (dt < ' ') || (lq > 99);

                if (!done)
                    cadd(qstr_p, dt);

            } while (!done);
        }

        if (!entry)
            err = message(dico,
                          "\"%s\" not evaluated.%s\n", t,
                          nolookup ? " Lookup failure." : "");
    } else {
        u = formula(dico, t, t + strlen(t), &err);
        numeric = 1;
    }

    if (numeric) {
        /* we want *exactly* 25 chars, we have
         *   sign, leading digit, '.', 'e', sign, upto 3 digits exponent
         * ==> 8 chars, thus we have 17 left for precision
         * don't print a leading '+', something choked
         */

        char buf[ACT_CHARACTS + 1];
        if (snprintf(buf, sizeof(buf), "% 25.17e", u) != ACT_CHARACTS) {
            fprintf(stderr, "ERROR: xpressn.c, %s(%d)\n", __FUNCTION__, __LINE__);
            controlled_exit(1);
        }
        scopys(qstr_p, buf);
    }

    return err;
}


/********* interface functions for spice3f5 extension ***********/

static int
insertnumber(dico_t *dico, int i, char *s, SPICE_DSTRINGPTR ustr_p)
/* insert u in string s in place of the next placeholder number */
{
    const char *u = spice_dstring_value(ustr_p);

    char buf[ACT_CHARACTS+1];

    long id = 0;
    int  n  = 0;

    char *p = strstr(s+i, "numparm__________");

    if (p &&
        (1 == sscanf(p, "numparm__________%8lx%n", &id, &n)) &&
        (n == ACT_CHARACTS) &&
        (id > 0) && (id < dynsubst + 1) &&
        (snprintf(buf, sizeof(buf), "%-25s", u) == ACT_CHARACTS))
    {
        memcpy(p, buf, ACT_CHARACTS);
        return (int)(p - s) + ACT_CHARACTS;
    }

    message
        (dico,
         "insertnumber: fails.\n"
         "  s+i = \"%s\" u=\"%s\" id=%ld\n",
         s+i, u, id);

    /* swallow everything on failure */
    return i + (int) strlen(s+i);
}


bool
nupa_substitute(dico_t *dico, char *s, char *r, bool err)
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
                err = message(dico, "Closing \"}\" not found.\n");
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
                err = message(dico, "Cannot compute substitute\n");

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
                    err = message(dico, "Closing \")\" not found.\n");
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
                message(dico, "Cannot compute &(expression)\n");
        }
    }

    spice_dstring_free(&qstr);
    spice_dstring_free(&tstr);

    return err;
}


static void
getword(char *s, SPICE_DSTRINGPTR tstr_p, int after, int *pi)
/* isolate a word from s after position "after". return i= last read+1 */
{
    int i = *pi;
    int ls;

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

    *pi = i;
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

        } while (!(cpos (c, ",;)}") >= 0)); /* legal separators */

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
nupa_assignment(dico_t *dico, char *s, char mode)
/* is called for all 'Param' lines of the input file.
   is also called for the params: section of a subckt .
   mode='N' define new local variable, else global...
   bug: we cannot rely on the transformed line, must re-parse everything!
*/
{
    /* s has the format: ident = expression; ident= expression ...  */
    int i, ls;
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

    while ((i < ls) && (s[i] <= ' '))
        i++;

    if (s[i] == Intro)
        i++;

    if (s[i] == '.')            /* skip any dot keyword */
        while (s[i] > ' ')
            i++;

    while ((i < ls) && !error) {

        getword(s, &tstr, i, &i);
        t_p = spice_dstring_value(&tstr);
        if (t_p[0] == '\0')
            error = message(dico, " Identifier expected\n");

        if (!error) {
            /* assignment expressions */
            while ((i <= ls) && (s[i - 1] != '='))
                i++;

            if (i > ls)
                error = message(dico, " = sign expected.\n");

            dtype = getexpress(s, &ustr, &i);

            if (dtype == 'R') {
                const char *tmp = spice_dstring_value(&ustr);
                rval = formula(dico, tmp, tmp + strlen(tmp), &error);
                if (error)
                    message(dico,
                            " Formula() error.\n"
                            "      %s\n", s);
            } else if (dtype == 'S') {
                wval = i;
            }

            err = nupa_define(dico, spice_dstring_value(&tstr), mode /* was ' ' */ ,
                         dtype, rval, wval, NULL);
            error = error || err;
        }

        if ((i < ls) && (s[i - 1] != ';'))
            error = message(dico, " ; sign expected.\n");
        /* else
           i++; */
    }

    spice_dstring_free(&tstr);
    spice_dstring_free(&ustr);

    return error;
}


bool
nupa_subcktcall(dico_t *dico, char *s, char *x, bool err)
/* s= a subckt define line, with formal params.
   x= a matching subckt call line, with actual params
*/
{
    int n, i, j, found_j, k, g, h, narg = 0, ls, nest;
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
        err = message(dico, " ! a subckt line!\n");
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
            while (--kptr >= optr && isspace_c(*kptr))
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
                message(dico, "identifier expected.\n");
            }
        }
    }

    /***** next, analyze the circuit call line */
    if (!err) {

        narg = 0;

        scopy_up(&tstr, x);
        j = 0;

        ls = spice_dstring_length(&tstr);

        spice_dstring_init(&parsebuf);
        scopyd(&parsebuf, &tstr);
        buf = spice_dstring_value(&parsebuf);

        found = found_j = 0;
        token = strtok(buf, " "); /* a bit more exact - but not sufficient everytime */
        j = j + (int) strlen(token) + 1;
        if (strcmp(token, spice_dstring_value(&subname)))
            while ((token = strtok(NULL, " ")) != NULL) {
                if (!strcmp(token, spice_dstring_value(&subname))) {
                    found = 1;
                    found_j = j;
                }
                j = j + (int) strlen(token) + 1;
            }

        j = found_j; /* last occurence of subname in buf */
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
                    if (t_p[k] > ' ')
                        message(dico, "Subckt call, symbol %c not understood\n", t_p[k]);
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
            message(dico, "Cannot find called subcircuit\n");
        }
    }

    /***** finally, execute the multi-assignment line */
    dicostack_push(dico);      /* create local symbol scope */

    if (narg != n) {
        err = message(dico,
                      " Mismatch: %d formal but %d actual params.\n"
                      "%s\n",
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
nupa_subcktexit(dico_t *dico)
{
    dicostack_pop(dico);
}
