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
#include "ngspice/stringskip.h"


/* random numbers in /maths/misc/randnumb.c */
#include "ngspice/randnumb.h"

/************ keywords ************/

extern long dynsubst;           /* see inpcom.c */

#define ACT_CHARACTS 25      /* actual string length to be inserted and replaced */

#define  S_init   0
#define  S_atom   1
#define  S_binop  2
#define  S_unop   3
#define  S_stop   4

static char* sort_idlist(char *list);

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
    return (nominal_val + stdvar * gauss1());
}


static double
gauss(double nominal_val, double rel_variation, double sigma)
{
    double stdvar;
    stdvar = nominal_val * rel_variation / sigma;
    return (nominal_val + stdvar * gauss1());
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
    "sqr sqrt sin cos exp ln arctan abs pow pwr max min int log log10 sinh cosh"
    " tanh ternary_fcn agauss sgn gauss unif aunif limit ceil floor"
    " asin acos atan asinh acosh atanh tan nint";


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

    if (dico->srcline >= 0) {
        if (ft_ngdebug) {
            fprintf
            (stderr,
                "Netlist line no. %d, new internal line no. %d:\n",
                dico->oldline, dico->srcline);
        }
        else {
            fprintf
            (stderr,
                "Netlist line no. %d:\n",
                dico->oldline);
        }
    }
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

    dico->srcline = -1;
    dico->errcount = 0;

    dico->symbols = TMALLOC(NGHASHPTR, asize);
    dico->inst_name = TMALLOC(char*, asize);
    dico->max_stack_depth = asize;
    dico->stack_depth = 0;              /* top of the stack */

    dico->symbols[0] = nghash_init(NGHASH_MIN_SIZE);
    nghash_unique(dico->symbols[0], TRUE); /* no rewrite of global symbols */

    dico->inst_symbols = NULL;          /* instance qualified are lazily allocated */

    if (newcompat.hs)
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
dicostack_push(dico_t *dico, char *inst_name)
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
    dico->inst_name[dico->stack_depth] = inst_name;
}


static void
dicostack_pop(dico_t *dico)
/* pop operation for nested subcircuit locals */
{
    char *inst_name;            /* name of subcircuit instance */
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
        /* build a qualified name */
        DS_CREATE(param_name, 100);

        NGHASH_FIRST(&iter);
        for (entry = (entry_t *) nghash_enumerateRE(htable_p, &iter);
                 entry;
                 entry = (entry_t *) nghash_enumerateRE(htable_p, &iter)) {
            ds_clear(&param_name);
            if (ds_cat_printf(&param_name, "%s.%s",
                    inst_name, entry->symbol) != DS_E_OK) {
                fprintf(stderr, "Error: DS could not add string %s\n", inst_name);
                controlled_exit(-1);
            }
            nupa_copy_inst_entry(ds_get_buf(&param_name), entry);
            dico_free_entry(entry);
        }
        nghash_free(htable_p, NULL, NULL);
        ds_free(&param_name);
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
entry_t *
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


static double
fetchnumentry(dico_t *dico, char *s, bool *perr)
{
    entry_t *entry = entrynb(dico, s);

    if (entry && (entry->tp == NUPA_REAL))
        return entry->vl;

    *perr = message(dico, "Undefined parameter [%s]\n", s);
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
        (entry->level < dico->stack_depth) && (entry->tp != NUPA_UNKNOWN))
    {
        entry = NULL;
    }

    if (!entry) {
        entry = TMALLOC(entry_t, 1);
        entry->symbol = copy(t);
        entry->tp = NUPA_UNKNOWN;      /* signal Unknown */
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
        if (entry->sbbase)
            tfree(entry->sbbase);
        tfree(entry);
    }
}


static bool
nupa_define(dico_t *dico,
       char *t,                 /* identifier to define */
       char op,                 /* option */
       nupa_type tpe,           /* type marker */
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
    nupa_type c;
    bool warn;
    entry_t *entry;             /* spice table entry */
    NGHASHPTR htable_p;         /* hash table */

    /* can't be lazy anymore */
    if (!(dico->symbols[dico->stack_depth]))
        dico->symbols[dico->stack_depth] = nghash_init(NGHASH_MIN_SIZE);

    htable_p = dico->symbols[dico->stack_depth];

    entry = attrib(dico, htable_p, t, op);

    if (!entry)
        return message(dico, " Symbol table overflow\n");

    c = entry->tp;

    if (c != NUPA_SUBCKT) {

        entry->vl = z;
        entry->tp = tpe;
        entry->ivl = w;
        entry->sbbase = base;
        /* if ((c != '?') && (i <= dico->stack[dico->tos])) { */
        if (c == NUPA_UNKNOWN)
            entry->level = dico->stack_depth; /* promote! */

        /* warn about re-write to a global scope! */
        if (entry->level < dico->stack_depth)
            warn = message(dico, "%s:%d overwritten.\n", t, entry->level);

    } else {
        /* error message for redefinition of subcircuits */
        if (0)
            message(dico, "subckt %s is already used,\n cannot be redefined\n", t);
    }

    return 0;
}


bool
defsubckt(dico_t *dico, const struct card *card)
/* called on 1st pass of spice source code,
   to enter subcircuit names
*/
{
    const char *s = card->line;
    int w = card->linenum;

    bool err;

    const char *s_end;

    while (*s && (*s != '.'))
        s++;                    /* skip 1st dotword */

    while (*s && ((unsigned char) (*s) > ' '))
        s++;

    while (*s && ((unsigned char) (*s) <= ' '))
        s++;                    /* skip blank */

    s_end = s;

    while (*s_end && ((unsigned char) (*s_end) > ' '))
        s_end++;

    if (s_end > s) {
        DS_CREATE(ustr, 200); /* temp user string */
        pscopy(&ustr, s, s_end);
        err = nupa_define(dico, ds_get_buf(&ustr), ' ',
                NUPA_SUBCKT, 0.0, w, NULL);
        ds_free(&ustr);
    } else {
        err = message(dico, "Subcircuit or Model without name.\n");
    }

    return err;
}


int
findsubckt(dico_t *dico, const char *s)
/* input: s is a subcircuit invocation line.
   returns 0 if not found, else the stored definition line number value */
{
    const char *name_e = skip_back_ws(s + strlen(s), s);
    const char *name_b = skip_back_non_ws(name_e, s);

    entry_t *entry;             /* symbol table entry */

    DS_CREATE(ustr, 200); /* u= subckt name is last token in string s */

    pscopy(&ustr, name_b, name_e);
    entry = entrynb(dico, ds_get_buf(&ustr));
    ds_free(&ustr);

    if (entry && (entry->tp == NUPA_SUBCKT)) {
        return entry->ivl;
    } else {
        message(dico, "Cannot find subcircuit.\n");
        return 0;
    }
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
        while ((p < s_end) && (*p == *keys))
            p++, keys++;
        if ((p >= s_end) && ((unsigned char) (*keys) <= ' '))
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
    case 'M':  return ciprefix("MEG", s) ? 1e6 : 1e-3;
    case 'U':  return 1e-6;
    case 'N':  return 1e-9;
    case 'P':  return 1e-12;
    case 'F':  return 1e-15;
    case 'A':  return 1e-18;
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
    } else if (strchr("=<>#GL", c)) {
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
        if ((unsigned char) c > ' ')
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
        x = x / y;
        break;
    case '^':                   /* power */
        if (newcompat.hs) {
            if (x < 0)
                x = pow(x, round(y));
            else if (x == 0)
                x = 0;
            else
                x = pow(x, y);
        }
        else if (newcompat.lt) {
            if (x >= 0)
                x = pow(x, y);
            else {
                /* If arg2 is quasi an integer, round it to have pow not fail
                   when arg1 is negative. Takes into account the double
                   representation which sometimes differs in the last digit(s). */
                if (AlmostEqualUlps(nearbyint(y), y, 10))
                    x = pow(x, round(y));
                else
                    /* As per LTSPICE specification for ** */
                    x = 0;
            }
        }
        else {
            x = pow(fabs(x), y);
        }
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
        t = trunc(x / y);
        x = x - y * t;
        break;
    case '\\':                  /* / */
        x = trunc(fabs(x / y));
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
    const char *s_orig = s;

    DS_CREATE(tstr, 200);

    for (i = 0; i <= nprece; i++) {
        accu[i] = 0.0;
        oper[i] = ' ';
    }

    /* trim trailing whitespace */
    while ((s_end > s) && ((unsigned char) (s_end[-1]) <= ' '))
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

            level = 1;
            for (; kptr < s_end; kptr++)
            {
                char d = *kptr;

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

                if ((d == ')') && (level <= 0)) {
                    break;
                }
            }

            // fixme, here level = 0 !!!!! (almost)

            if (kptr >= s_end) {
                error = message(dico, "Closing \")\" not found.\n");
                natom++;        /* shut up other error message */
            } else {
                if (arg2 >= s) {
                    v = formula(dico, s, arg2, &error);
                    s = arg2 + 1;
                }
                if (arg3 >= s) {
                    w = formula(dico, s, arg3, &error);
                    s = arg3 + 1;
                }
                u = formula(dico, s, kptr, &error);
                state = S_atom;
                if (fu > 0) {
                    if (fu == XFU_TERNARY_FCN)
                        u = ternary_fcn(v, w, u);
                    else if (fu == XFU_AGAUSS)
                        u = agauss(v, w, u);
                    else if (fu == XFU_GAUSS)
                        u = gauss(v, w, u);
                    else if (fu == XFU_UNIF)
                        u = unif(v, u);
                    else if (fu == XFU_AUNIF)
                        u = aunif(v, u);
                    else if (fu == XFU_LIMIT)
                        u = limit(v, u);
                    else
                        u = mathfunction(fu, v, u);
                }
            }
            s = kptr + 1;
            fu = 0;
        } else if (alfa(c)) {
            const char *s_next = fetchid(s, s_end);
            fu = keyword(fmathS, s, s_next); /* numeric function? */
            if (fu > 0) {
                state = S_init;  /* S_init means: ignore for the moment */
            } else {
                ds_clear(&tstr);
                pscopy(&tstr, s, s_next);
                u = fetchnumentry(dico, ds_get_buf(&tstr), &error);
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

    ds_free(&tstr);

    if (error)
        return 1.0;
    else
        return accu[topop];
}


/* Check for a string expression, return end  pointer or NULL.
 * A string expression is a sequence of quoted strings and string
 * variables, optionally enclosed by '{}' with no interventing space.
 * If successful return pointer to next char, otherwise NULL.
 * Evaluated string is returned in *qstr_p (may be NULL).
 */

static char *string_expr(dico_t *dico, DSTRINGPTR qstr_p,
                         const char *t, const char *t_end)
{
    const char *tie;
    bool        ok = FALSE;

    while (isblank(*t) && t < t_end)
        ++t;
    if (qstr_p)
        ds_clear(qstr_p);
    for (; t < t_end; ) {
        if (*t == '"') {
            /* String constant. */

            tie = ++t;
            while (*t != '"' && t < t_end)
                ++t;
            if (qstr_p)
                pscat(qstr_p, tie, t);
            if (*t == '"')
                ++t;
            ok = TRUE;
            continue;
        }
        if (*t == '{') {
            /* Isolate and check wrapped identifier. */

            tie = ++t;
            while (t < t_end) {
                if (*t == '}')
                    break;
                ++t;
            }
        } else {
            /* Last option: naked string-valued param. */

            tie = t;
            t = fetchid(t, t_end);
            if (t == tie )
                return NULL;
        }
        /* Now pointers tie, t should bracket an identifier. */

        {
            DS_CREATE(lcl_str, 200);
            entry_t    *entry;

            /* Formula is a single identifier. */

            pscopy(&lcl_str, tie, t);
            entry = entrynb(dico, ds_get_buf(&lcl_str));
            ds_free(&lcl_str);
            if (entry && (entry->tp == NUPA_STRING)) {
                if (qstr_p)
                    pscat(qstr_p, entry->sbbase, NULL);
                ok = TRUE;
            } else {
                return NULL;
            }
        }
        if (*t == '}')
            ++t;
    }
    return ok ? (char *)t : NULL;
}

/* stupid, produce a string representation of a given double
 *   to be spliced back into the circuit deck
 * we want *exactly* 25 chars, we have
 *   sign, leading digit, '.', 'e', sign, upto 3 digits exponent
 * ==> 8 chars, thus we have 17 left for precision
 * don't print a leading '+', something choked
 */

static void
double_to_string(DSTRINGPTR qstr_p, double value)
{
    char buf[ACT_CHARACTS + 1];
    if (snprintf(buf, sizeof(buf), "% 23.15e  ", value) != ACT_CHARACTS) {
        fprintf(stderr, "ERROR: xpressn.c, %s(%d)\n", __FUNCTION__, __LINE__);
        controlled_exit(1);
    }
    scopys(qstr_p, buf);
}


/* transform exression in string `t' to result q */
static bool
evaluate_expr(dico_t *dico, DSTRINGPTR qstr_p, const char *t, const char * const t_end)
{
    bool err = 0;
    double u;

    ds_clear(qstr_p);
    if (string_expr(dico, qstr_p, t, t_end))
        return 0;
    u = formula(dico, t, t_end, &err);
    if (err)
        return err;

    double_to_string(qstr_p, u);

    return 0;
}


/********* interface functions for spice3f5 extension ***********/

static bool
insertnumber(dico_t *dico, char **lp, DSTRINGPTR ustr_p)
/* insert *ustr_p in string *lp in place of the next placeholder number */
{
    const char *u = ds_get_buf(ustr_p);
    char       *s = *lp; // Point to line contents
    long        id = 0;
    int         n;

    char *p = strstr(s, "numparm__________");

    if (p &&
        (1 == sscanf(p, "numparm__________%8lx%n", &id, &n)) &&
        (n == ACT_CHARACTS) &&
        (id > 0) && (id < dynsubst + 1)) {
        /* Found a target for substitution. */

        n = (int)ds_get_length(ustr_p);
        if (n <= ACT_CHARACTS) {
            char buf[ACT_CHARACTS + 1];

            /* Replace in place. */

            snprintf(buf, sizeof buf, "%-*s", ACT_CHARACTS, u);
            memcpy(p, buf, ACT_CHARACTS);
        } else {
            char *newline;

            /* Requires reallocation. */

            newline = malloc((p - s) + n + strlen(p + ACT_CHARACTS) + 1);
            if (!newline) {
                message(dico, "nupa_substitute failed: no memory\n");
                return TRUE;
            }
            memcpy(newline, s, (p - s));
            memcpy(newline + (p - s), u, n);
            strcpy(newline + (p - s) + n, p + ACT_CHARACTS);
            free(*lp);
            *lp = newline;
        }
        return FALSE;
    }

    message
        (dico,
         "insertnumber: fails.\n"
         "  s = \"%s\" u=\"%s\" id=%ld\n",
         s, u, id);
    return TRUE;
}


bool
nupa_substitute(dico_t *dico, const char *s, char **lp)
/* s: pointer to original source line.
   lp: pointer to result line pointer, line already heavily modified wrt s:
   anywhere we find a 25-char numstring in *lp, substitute it.
*/
{
    const char * const s_end = s + strlen(s);
    bool err = 0;

    DS_CREATE(qstr, 200); /* temp result dynamic string */

    while (s < s_end) {
        char c = *s++;

        if (c == '{') {
            /* try ps expression syntax */
            const char *kptr = s;
            int nnest = 1;

            for (; *kptr; kptr++) {

                char d = *kptr;

                if (d == '{')
                    nnest++;
                else if (d == '}')
                    nnest--;

                if (nnest == 0)
                    break;
            }

            if (*kptr == '\0') {
                err = message(dico, "Closing \"}\" not found in line fragment\n    {%s.\n", s);
                goto Lend;
            }

            /* exeption made for .meas */
            if (s + 4 == kptr && strncasecmp(s, "LAST", 4) == 0) {
                ds_clear(&qstr);
                sadd(&qstr, "last");
            } else {
                err = evaluate_expr(dico, &qstr, s, kptr);
                if (err) {
                    err = message(dico, "Cannot compute substitute\n");
                    goto Lend;
                }
            }

            s = kptr + 1;
            err = insertnumber(dico, lp, &qstr);
            if (err)
                break;
        }
    }

 Lend:
    ds_free(&qstr);
    return err;
}


static const char *
getword(const char *s, DSTRINGPTR tstr_p)
{
    const char *s_end = s + strlen(s);
    const char *word;

    while ((s < s_end - 1) && !alfa(*s))
        s++;

    word = s;
    while (alfa(*s) || isdigit_c(*s))
        s++;

    ds_clear(tstr_p);
    pscopy(tstr_p, word, s);

    return s;
}


static char *
getexpress(dico_t *dico, nupa_type *type, DSTRINGPTR tstr_p, const char *s)
/* returns expression-like string until next separator
   Input  i=position before expr, output  i=just after expr, on separator.
   returns tpe=='R' if (numeric, 'S' if (string only
*/
{
    const char *s_end = s + strlen(s);
    const char *p;
    nupa_type tpe;

    while ((s < s_end - 1) && ((unsigned char)(* s) <= ' '))
        s++;                    /*white space ? */

    /* Check for injected semicolon separator in assignment list. */
    p = strchr(s, ';');
    if (p)
        s_end = p;

    p = string_expr(dico, NULL, s, s_end);
    if (p) {
        tpe = NUPA_STRING;
    } else {

        if (*s == '{')
            s++;
        p = s;

        for (; p < s_end; p++) {

            if (strchr(",;)}", *p)) /* legal separators */
                break;

            if (*p == '(') {
                /* sub-formula */
                int level = 1;
                p++;
                for (; p < s_end; p++) {

                    char d = *p;

                    if (d == '(')
                        level++;
                    else if (d == ')')
                        level--;

                    if (level <= 0)
                        break;
                }
            }
        }
        tpe = NUPA_REAL;
    }

    pscopy(tstr_p, s, p);

    if (*p == '}')
        p++;

    if (type)
        *type = tpe;

    return (char *) p;
}


bool
nupa_assignment(dico_t *dico, const char *s, char mode)
/* is called for all 'Param' lines of the input file.
   is also called for the params: section of a subckt .
   mode='N' define new local variable, else global...
   bug: we cannot rely on the transformed line, must re-parse everything!
*/
{
    if (!s || !*s)
        return 1;
    /* s has the format: ident = expression; ident= expression ...  */
    const char * const s_end = s + strlen(s);
    const char *p = s;
    const char *tmp;
    char       *sval = NULL;
    bool error = 0;
    nupa_type dtype;
    int wval = 0;
    double rval = 0.0;
    char *t_p;                  /* dstring contents value */

    DS_CREATE(tstr, 200); /* temporary dstrings */
    DS_CREATE(ustr, 200);

    while ((p < s_end) && ((unsigned char) (*p) <= ' '))
        p++;

    if (*p == '.')            /* skip any dot keyword */
        while ((unsigned char) (*p) > ' ')
            p++;

    while (p < s_end) {

        p = getword(p, &tstr);
        t_p = ds_get_buf(&tstr);
        if (t_p[0] == '\0') {
            error = message(dico, " Identifier expected\n");
            break;
        }

        /* assignment expressions */
        while ((p < s_end) && (*p != '='))
            p++;

        if (p >= s_end) {
            error = message(dico, " = sign expected.\n");
            break;
        }

        p = getexpress(dico, &dtype, &ustr, p + 1) + 1;

        tmp = ds_get_buf(&ustr);
        if (dtype == NUPA_REAL) {
            rval = formula(dico, tmp, tmp + strlen(tmp), &error);
            if (error) {
                message(dico,
                        " Formula() error.\n"
                        "      |%s| : |%s|=|%s|\n", s, ds_get_buf(&tstr), ds_get_buf(&ustr));
                break;
            }
        } else if (dtype == NUPA_STRING) {
            DS_CREATE(sstr, 200);
            string_expr(dico, &sstr, tmp, tmp + strlen(tmp));
            sval = copy(ds_get_buf(&sstr));
            ds_free(&sstr);
        }

        error = nupa_define(dico, ds_get_buf(&tstr), mode /* was ' ' */ ,
                            dtype, rval, wval, sval);
        if (error)
            break;

        if ((p < s_end) && (p[-1] != ';')) {
            error = message(dico, " ; sign expected.\n");
            break;
        }
    }

    ds_free(&tstr);
    ds_free(&ustr);

    return error;
}


/*
 * search for whitespace delimited occurence of `identifier' in str
 *   delimited by whitespace or begin/end of string
 */

static char *
search_isolated_identifier(char *str, const char *identifier)
{
    char *str_begin = str;

    while ((str = strstr(str, identifier)) != NULL) {

        if (str <= str_begin || isspace_c(str[-1])) {
            char after = str[strlen(identifier)];
            if (!after || isspace_c(after))
                return str;
        }

        str += strlen(identifier);
    }

    return NULL;
}


bool
nupa_subcktcall(dico_t *dico, const char *s, const char *x,
        char *inst_name)
/* s= a subckt define line, with formal params.
   x= a matching subckt call line, with actual params
*/
{
    int n, narg = 0;
    bool err = 0;

    DS_CREATE(subname, 200);
    DS_CREATE(tstr, 200);
    DS_CREATE(ustr, 200);
    DS_CREATE(vstr, 200);
    DS_CREATE(idlist, 200);

    /***** first, analyze the subckt definition line */
    n = 0; /* number of parameters if any */
    scopys(&tstr, s);

    /* Get the subcircuit name in subname. */

    const char *j2 = strstr(ds_get_buf(&tstr), "subckt");
    if (j2) {
        j2 = skip_ws(j2 + 6);     /* skip subckt and whitespace */
        pscopy(&subname, j2, skip_non_ws(j2));
    } else {
        err = message(dico, " ! a subckt line!\n");
    }

    /* Scan the .subckt line for assignments, copying templates to idlist. */

    const char *i2 = strstr(ds_get_buf(&tstr), "params:");

    if (i2) {
        const char* optr, * jptr;

        /* search identifier to the left of '=' assignments */
        for (optr = i2 + 7;
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

        /*
          skip over instance name -- fixes bug where instance 'x1' is
          same name as subckt 'x1'
        */
        scopys(&tstr, skip_non_ws(x));

        char * const t_p = ds_get_buf(&tstr);
        char *jp = NULL;

        /* Search for the last occurence of `subname' in the call line. */
        for (;;) {
            char *next_p = search_isolated_identifier(jp ? jp + 1 : t_p,
                    ds_get_buf(&subname));
            if (!next_p)
                break;
            jp = next_p;
        }

        if (jp) {
            jp += ds_get_length(&subname);

            while (isspace_c(*jp) || (*jp == ','))
                jp++;

            /* jp is pointing to the 1st position of arglist now */

            while (*jp) {
                /* try to fetch valid arguments */
                char *kp = jp;
                ds_clear(&ustr);

                if (alfanum(*kp) || *kp == '.') {
                    /* number, identifier */
                    jp = skip_non_ws(kp);
                    pscopy(&ustr, kp, jp);
                } else if (*kp == '{') {
                    jp = getexpress(dico, NULL, &ustr, jp);
                } else {
                    jp++;
                    if ((unsigned char)(*kp) > ' ') {
                        fprintf(stderr, "Error in line: %s\n", x);
                        fprintf(stderr, "    near %s\n", kp);
                        message(dico, "Subckt call, symbol %c not understood\n\n", *kp);

                    }
                }

                /* Substitute the parameter for one of the '$' characters
                 * in idlist.
                 */
                char * const u_p = ds_get_buf(&ustr);
                if (*u_p) {
                    char * const idlist_p = ds_get_buf(&idlist);
                    char *dollar = strchr(idlist_p, '$');
                    if (dollar) {
                        /* replace dollar with expression string u */
                        pscopy(&vstr, idlist_p, dollar);
                        sadd(&vstr, u_p);
                        sadd(&vstr, dollar + 1);
                        scopyd(&idlist, &vstr);
                    }
                    narg++;
                }
            }
        } else {
            message(dico, "Cannot find called subcircuit\n");
        }
    }

    /***** finally, execute the multi-assignment line */
    dicostack_push(dico, inst_name);      /* create local symbol scope */

    if (narg != n) {
        err = message(dico,
                      " Mismatch: %d formal but %d actual params.\n"
                      "%s\n",
                      n, narg, ds_get_buf(&idlist));
        /* ;} else { debugwarn(dico, idlist) */
    }

    /* sort the idlist, so that plain numerical entries like nf=2 move to the front */
    char* sortedlist = sort_idlist(ds_get_buf(&idlist));

    err = nupa_assignment(dico, sortedlist, 'N');

    ds_free(&subname);
    ds_free(&tstr);
    ds_free(&ustr);
    ds_free(&vstr);
    ds_free(&idlist);

    tfree(sortedlist);

    return err;
}


void
nupa_subcktexit(dico_t *dico)
{
    dicostack_pop(dico);
}


struct nupa_type {                    /* used as a type-checked enum */
    const char *name;
};

const struct nupa_type S_nupa_real = { "NUPA_REAL" };
const struct nupa_type S_nupa_string = { "NUPA_STRING" };
const struct nupa_type S_nupa_subckt = { "NUPA_SUBCKT" };
const struct nupa_type S_nupa_unknown = { "NUPA_UNKNOWN" };

/* get the instance line list, sort numerical entries (eg. nf=1) to the front */
static char* sort_idlist(char* list) {
    wordlist* wl = NULL, *wle = NULL;
    bool start = TRUE;
    char* cut_list = list, *ret;
    while (*cut_list != '\0') {
        int error;
        char* token = gettok_char(&cut_list, ';', TRUE, FALSE);
        char* eqstr = strchr(token, '=');
        eqstr++;
        INPevaluate(&eqstr, &error, 1);
        /* num entry, prepend word */
        if (error == 0 && *eqstr == '\0') {
            wle = wl_cons(token, wle);
            if (start)
                wl = wle;
            start = FALSE;
        }
        /* expression, append word */
        else {
            wl_append_word(&wl, &wl, token);
            if (start)
                wle = wl;
            start = FALSE;
        }
    }
    ret = wl_flatten(wle);
    wl_free(wle);
    return ret;
}
