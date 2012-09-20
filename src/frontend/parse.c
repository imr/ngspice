/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Wayne A. Christopher, U. C. Berkeley CAD Group
**********/

/*
 * A simple operator-precedence parser for algebraic expressions.
 * This also handles relational and logical expressions.
 */

#include "ngspice/ngspice.h"
#include "ngspice/bool.h"
#include "ngspice/fteparse.h"
#include "ngspice/fteext.h"
#include "ngspice/sim.h"

#include "evaluate.h"
#include "parse.h"


static bool checkvalid(struct pnode *pn);
static struct pnode *mkbnode(int opnum, struct pnode *arg1, struct pnode *arg2);
static struct pnode *mkunode(int op, struct pnode *arg);
static struct pnode *mkfnode(const char *func, struct pnode *arg);
static struct pnode *mknnode(double number);
static struct pnode *mksnode(const char *string);

#include "parse-bison.c"

char *db_print_pnode_tree(struct pnode *p, char *print);


struct pnode *
ft_getpnames(wordlist *wl, bool check)
{
    struct pnode *pn;
    char *xsbuf, *sbuf;
    int rv;

    if (!wl) {
        fprintf(cp_err, "Warning: NULL arithmetic expression\n");
        return (NULL);
    }

    xsbuf = sbuf = wl_flatten(wl);

    rv = PPparse(&sbuf, &pn);

    tfree(xsbuf);

    if (rv)
        return (NULL);

    if (check && !checkvalid(pn))
        return (NULL);

    return (pn);
}


/* See if there are any variables around which have length 0 and are
 * not named 'list'. There should really be another flag for this...
 */

static bool
checkvalid(struct pnode *pn)
{
    while (pn) {
        if (pn->pn_value) {
            if ((pn->pn_value->v_length == 0) &&
                !eq(pn->pn_value->v_name, "list")) {
                if (eq(pn->pn_value->v_name, "all"))
                    fprintf(cp_err,
                            "Error: %s: no matching vectors.\n",
                            pn->pn_value->v_name);
                else
                    fprintf(cp_err,
                            "Error(parse.c--checkvalid): %s: no such vector.\n",
                            pn->pn_value->v_name);
                return (FALSE);
            }
        } else if (pn->pn_func || (pn->pn_op && (pn->pn_op->op_arity == 1))) {
            if (!checkvalid(pn->pn_left))
                return (FALSE);
        } else if (pn->pn_op && (pn->pn_op->op_arity == 2)) {
            if (!checkvalid(pn->pn_left))
                return (FALSE);
            if (!checkvalid(pn->pn_right))
                return (FALSE);
        } else {
            fprintf(cp_err,
                    "checkvalid: Internal Error: bad node\n");
        }
        pn = pn->pn_next;
    }
    return (TRUE);
}


/* Some auxiliary functions for building the parse tree. */

static struct op ops[] = {
    { PT_OP_PLUS,     "+",   2, {(void(*)(void)) op_plus} },
    { PT_OP_MINUS,    "-",   2, {(void(*)(void)) op_minus} },
    { PT_OP_TIMES,    "*",   2, {(void(*)(void)) op_times} },
    { PT_OP_MOD,      "%",   2, {(void(*)(void)) op_mod} },
    { PT_OP_DIVIDE,   "/",   2, {(void(*)(void)) op_divide} },
    { PT_OP_COMMA,    ",",   2, {(void(*)(void)) op_comma} },
    { PT_OP_POWER,    "^",   2, {(void(*)(void)) op_power} },
    { PT_OP_EQ,       "=",   2, {(void(*)(void)) op_eq} },
    { PT_OP_GT,       ">",   2, {(void(*)(void)) op_gt} },
    { PT_OP_LT,       "<",   2, {(void(*)(void)) op_lt} },
    { PT_OP_GE,       ">=",  2, {(void(*)(void)) op_ge} },
    { PT_OP_LE,       "<=",  2, {(void(*)(void)) op_le} },
    { PT_OP_NE,       "<>",  2, {(void(*)(void)) op_ne} },
    { PT_OP_AND,       "&",  2, {(void(*)(void)) op_and} },
    { PT_OP_OR,       "|",   2, {(void(*)(void)) op_or} },
    { PT_OP_INDX,     "[",   2, {(void(*)(void)) op_ind} },
    { PT_OP_RANGE,    "[[",  2, {(void(*)(void)) op_range} },
    { PT_OP_TERNARY,  "?:",  2, {NULL} },
    { 0,               NULL, 0, {NULL} }
};


static struct op uops[] = {
    { PT_OP_UMINUS, "-",  1, {(void(*)(void)) op_uminus} },
    { PT_OP_NOT,    "~",  1, {(void(*)(void)) op_not} },
    { 0,            NULL, 0, {NULL} }
};


/* We have 'v' declared as a function, because if we don't then the defines
 * we do for vm(), etc won't work. This is caught in evaluate(). Bad kludge.
 */

typedef void* cx_function_t(void*, short int, int, int*, short int*);

struct func ft_funcs[] = {
    { "mag",         cx_mag },
    { "magnitude",   cx_mag },
    { "cph",         cx_cph },  /* SJdV */
    { "cphase",      cx_cph },  /* SJdV Continious phase*/
    { "ph",          cx_ph },
    { "phase",       cx_ph },
    { "j",           cx_j },
    { "real",        cx_real },
    { "re",          cx_real },
    { "imag",        cx_imag },
    { "im",          cx_imag },
    { "db",          cx_db },
    { "log",         cx_log },
    { "log10",       cx_log },
    { "ln",          cx_ln },
    { "exp",         cx_exp },
    { "abs",         cx_mag },
    { "sqrt",        cx_sqrt },
    { "sin",         cx_sin },
    { "cos",         cx_cos },
    { "tan",         cx_tan },
    { "sinh",        cx_sinh },
    { "cosh",        cx_cosh },
    { "tanh",        cx_tanh },
    { "atan",        cx_atan },
    { "norm",        cx_norm },
    { "rnd",         cx_rnd },
    { "sunif",       cx_sunif },
    { "poisson",     cx_poisson },
    { "exponential", cx_exponential },
    { "sgauss",      cx_sgauss },
    { "pos",         cx_pos },
    { "floor",       cx_floor },
    { "ceil",        cx_ceil },
    { "mean",        cx_mean },
    { "avg",         cx_avg }, /* A.Roldan 03/06/05 incremental average new function */
    { "group_delay", (cx_function_t*) cx_group_delay }, /* A.Roldan 10/06/05 group delay new function */
    { "vector",      cx_vector },
    { "unitvec",     cx_unitvec },
    { "length",      cx_length },
    { "vecmin",      cx_min },
    { "vecmax",      cx_max },
    { "vecd",        cx_d },
    { "interpolate", (cx_function_t*) cx_interpolate },
    { "deriv",       (cx_function_t*) cx_deriv },
    { "v",           NULL },
    { NULL,          NULL }
};

struct func func_uminus = { "minus", cx_uminus };

struct func func_not = { "not", cx_not };


/* Binary operator node. */

static struct pnode *
mkbnode(int opnum, struct pnode *arg1, struct pnode *arg2)
{
    struct op *o;
    struct pnode *p;

    for (o = &ops[0]; o->op_name; o++)
        if (o->op_num == opnum)
            break;

    if (!o->op_name)
        fprintf(cp_err, "mkbnode: Internal Error: no such op num %d\n",
                opnum);

    p = alloc(struct pnode);
    p->pn_use = 0;
    p->pn_value = NULL;
    p->pn_name = NULL;  /* sjb */
    p->pn_func = NULL;
    p->pn_op = o;
    p->pn_left = arg1;
    if (p->pn_left)
        p->pn_left->pn_use++;
    p->pn_right = arg2;
    if (p->pn_right)
        p->pn_right->pn_use++;
    p->pn_next = NULL;
    return (p);
}


/* Unary operator node. */

static struct pnode *
mkunode(int op, struct pnode *arg)
{
    struct pnode *p;
    struct op *o;

    p = alloc(struct pnode);
    for (o = uops; o->op_name; o++)
        if (o->op_num == op)
            break;

    if (!o->op_name)
        fprintf(cp_err, "mkunode: Internal Error: no such op num %d\n",
                op);

    p->pn_op = o;
    p->pn_use = 0;
    p->pn_value = NULL;
    p->pn_name = NULL;  /* sjb */
    p->pn_func = NULL;
    p->pn_left = arg;
    if (p->pn_left)
        p->pn_left->pn_use++;
    p->pn_right = NULL;
    p->pn_next = NULL;
    return (p);
}


/* Function node. We have to worry about a lot of things here. Something
 * like f(a) could be three things -- a call to a standard function, which
 * is easiest to deal with, a variable name, in which case we do the
 * kludge with 0-length lists, or it could be a user-defined function,
 * in which case we have to figure out which one it is, substitute for
 * the arguments, and then return a copy of the expression that it was
 * defined to be.
 */

static struct pnode *
mkfnode(const char *func, struct pnode *arg)
{
    struct func *f;
    struct pnode *p, *q;
    struct dvec *d;
    char buf[BSIZE_SP];

    (void) strcpy(buf, func);
    strtolower(buf);  /* Make sure the case is ok. */

    for (f = &ft_funcs[0]; f->fu_name; f++)
        if (eq(f->fu_name, buf))
            break;

    if (f->fu_name == NULL) {
        /* Give the user-defined functions a try. */
        q = ft_substdef(func, arg);
        if (q)
            return (q);
    }

    if ((f->fu_name == NULL) && arg->pn_value) {
        /* Kludge -- maybe it is really a variable name. */
        (void) sprintf(buf, "%s(%s)", func, arg->pn_value->v_name);
        d = vec_get(buf);
        if (d == NULL) {
            /* Well, too bad. */
            fprintf(cp_err, "Error: no such function as %s.\n",
                    func);
            return (NULL);
        }
        /* (void) strcpy(buf, d->v_name); XXX */
        return (mksnode(buf));
    } else if (f->fu_name == NULL) {
        fprintf(cp_err, "Error: no function as %s with that arity.\n",
                func);
        return (NULL);
    }

    if (!f->fu_func && arg->pn_op && arg->pn_op->op_num == PT_OP_COMMA) {
        p = mkbnode(PT_OP_MINUS, mkfnode(func, arg->pn_left),
                    mkfnode(func, arg->pn_right));
        tfree(arg);
        return p;
    }

    p = alloc(struct pnode);
    p->pn_use = 0;
    p->pn_name = NULL;
    p->pn_value = NULL;
    p->pn_func = f;
    p->pn_op = NULL;
    p->pn_left = arg;
    if (p->pn_left)
        p->pn_left->pn_use++;
    p->pn_right = NULL;
    p->pn_next = NULL;
    return (p);
}


/* Number node. */

static struct pnode *
mknnode(double number)
{
    struct pnode *p;
    struct dvec *v;
    char buf[BSIZE_SP];

    p = alloc(struct pnode);
    v = alloc(struct dvec);
    ZERO(v, struct dvec);
    p->pn_use = 0;
    p->pn_name = NULL;
    p->pn_value = v;
    p->pn_func = NULL;
    p->pn_op = NULL;
    p->pn_left = p->pn_right = NULL;
    p->pn_next = NULL;

    /* We don't use printnum because it screws up mkfnode above. We have
     * to be careful to deal properly with node numbers that are quite
     * large...
     */
    if (number < MAXPOSINT)
        (void) sprintf(buf, "%d", (int) number);
    else
        (void) sprintf(buf, "%G", number);
    v->v_name = copy(buf);
    v->v_type = SV_NOTYPE;
    v->v_flags = VF_REAL;
    v->v_realdata = TMALLOC(double, 1);
    *v->v_realdata = number;
    v->v_length = 1;
    v->v_plot = NULL;
    vec_new(v);
    return (p);
}


/* String node. */

static struct pnode *
mksnode(const char *string)
{
    struct dvec *v, *nv, *vs, *newv = NULL, *end = NULL;
    struct pnode *p;

    p = alloc(struct pnode);
    p->pn_use = 0;
    p->pn_name = NULL;
    p->pn_func = NULL;
    p->pn_op = NULL;
    p->pn_left = p->pn_right = NULL;
    p->pn_next = NULL;
    v = vec_get(string);
    if (v == NULL) {
        nv = alloc(struct dvec);
        ZERO(nv, struct dvec);
        p->pn_value = nv;
        nv->v_name = copy(string);
        return (p);
    }
    p->pn_value = NULL;

    /* It's not obvious that we should be doing this, but... */
    for (vs = v; vs; vs = vs->v_link2) {
        nv = vec_copy(vs);
        vec_new(nv);
        if (end)
            end->v_link2 = nv;
        else
            newv = end = nv;
        end = nv;
    }
    p->pn_value = newv;

    /* va: tfree v in case of @xxx[par], because vec_get created a new vec and
       nobody will free it elsewhere */
    /*if (v && v->v_name && *v->v_name == '@' && isreal(v) && v->v_realdata) {
      vec_free(v);
      } */
    /* The two lines above have been commented out to prevent deletion of @xxx[par]
       after execution of only a single command like plot @xxx[par] or write. We need to
       monitor if this will lead to excessive memory usage. h_vogt 090221 */
    return (p);
}


/* Don't call this directly, always use the free_pnode() macro.
   The linked pnodes do not necessarily form a perfect tree as some nodes get
   reused.  Hence, in this recursive walk trough the 'tree' we only free node
   that have their pn_use value at zero. Nodes that have pn_use values above
   zero have the link severed and their pn_use value decremented.
   In addition, we don't walk past nodes with pn_use values avoid zero, just
   in case we have a circular reference (this probable does not happen in
   practice, but it does no harm playing safe) */
void
free_pnode_x(struct pnode *t)
{
    if (!t)
        return;

    /* don't walk past nodes used elsewhere. We decrement the pn_use value here,
       but the link gets severed by the action of the free_pnode() macro */
    if (t->pn_use > 1) {
        t->pn_use--;
    } else {
        /* pn_use is now 1, so its safe to free the pnode */
        free_pnode(t->pn_left);
        free_pnode(t->pn_right);
        free_pnode(t->pn_next);
        tfree(t->pn_name); /* va: it is a copy() of original string, can be free'd */
        if (t->pn_value && !(t->pn_value->v_flags & VF_PERMANENT))
            vec_free(t->pn_value); /* patch by Stefan Jones */
        tfree(t);
    }
}


/* here is the original free_node, which is needed in spec.c and com_fft.c */
void
free_pnode_o(struct pnode *t)
{
    if (!t)
        return;
    free_pnode(t->pn_left);
    free_pnode(t->pn_right);
    free_pnode(t->pn_next);
    tfree(t);
}


static void
db_print_func(FILE *fdst, struct func *f)
{
    if (!f) {
        fprintf(fdst, "nil");
        return;
    }

    fprintf(fdst, "(func :fu_name %s :fu_func %p)", f->fu_name, f->fu_func);
}


static void
db_print_op(FILE *fdst, struct op *op)
{
    if (!op) {
        fprintf(fdst, "nil");
        return;
    }

    fprintf(fdst, "(op :op_num %d :op_name %s :op_arity %d :op_func %p)",
            op->op_num, op->op_name, op->op_arity, op->op_func.anonymous);
}


static void
db_print_dvec(FILE *fdst, struct dvec *d)
{
    if (!d) {
        fprintf(fdst, "nil");
        return;
    }

    fprintf(fdst, "(dvec :v_name %s :v_type %d :v_flags %d :v_length %d ...)",
            d->v_name, d->v_type, d->v_flags, d->v_length);
}


static void
db_print_pnode(FILE *fdst, struct pnode *p)
{
    if (!p) {
        fprintf(fdst, "nil\n");
        return;
    }

    if (!p->pn_name && p->pn_value && !p->pn_func && !p->pn_op &&
        !p->pn_left && !p->pn_right && !p->pn_next)
    {
        fprintf(fdst, "(pnode-value :pn_use %d", p->pn_use);
        fprintf(fdst, " :pn_value "); db_print_dvec(fdst, p->pn_value);
        fprintf(fdst, ")\n");
        return;
    }

    if (!p->pn_name && !p->pn_value && p->pn_func && !p->pn_op &&
        !p->pn_right && !p->pn_next)
    {
        fprintf(fdst, "(pnode-func :pn_use %d", p->pn_use);
        fprintf(fdst, "\n :pn_func "); db_print_func(fdst, p->pn_func);
        fprintf(fdst, "\n :pn_left "); db_print_pnode(fdst, p->pn_left);
        fprintf(fdst, ")\n");
        return;
    }

    if (!p->pn_name && !p->pn_value && !p->pn_func && p->pn_op &&
        !p->pn_next)
    {
        fprintf(fdst, "(pnode-op :pn_use %d", p->pn_use);
        fprintf(fdst, "\n :pn_op "); db_print_op(fdst, p->pn_op);
        fprintf(fdst, "\n :pn_left "); db_print_pnode(fdst, p->pn_left);
        fprintf(fdst, "\n :pn_right "); db_print_pnode(fdst, p->pn_right);
        fprintf(fdst, ")\n");
        return;
    }

    fprintf(fdst, "(pnode :pn_name \"%s\" pn_use %d", p->pn_name, p->pn_use);
    fprintf(fdst, "\n :pn_value "); db_print_dvec(fdst, p->pn_value);
    fprintf(fdst, "\n :pn_func "); db_print_func(fdst, p->pn_func);
    fprintf(fdst, "\n :pn_op "); db_print_op(fdst, p->pn_op);
    fprintf(fdst, "\n :pn_left "); db_print_pnode(fdst, p->pn_left);
    fprintf(fdst, "\n :pn_right "); db_print_pnode(fdst, p->pn_right);
    fprintf(fdst, "\n :pn_next "); db_print_pnode(fdst, p->pn_next);
    fprintf(fdst, "\n)\n");
}


char *
db_print_pnode_tree(struct pnode *p, char *print)
{
#if 1
    NG_IGNORE(print);
    db_print_pnode(stdout, p);
    return NULL;
#else
    char *buf;
    size_t  buf_size;
    FILE *db_stream = open_memstream(&buf, &buf_size);
    db_print_pnode(db_stream, p);
    fclose(db_stream);
    if (print)
        printf("%s:%d: %s {%s}\n%s\n", __FILE__, __LINE__, __func__, print, buf);
    return buf;
#endif
}


int
PPlex(YYSTYPE *lvalp, struct PPltype *llocp, char **line)
{
    static char *specials = " \t%()-^+*,/|&<>~=";
    char  *sbuf = *line;
    int token;

    while ((*sbuf == ' ') || (*sbuf == '\t'))
        sbuf++;

    llocp->start = sbuf;

#define lexer_return(token_, length)                            \
    do { token = token_; sbuf += length; goto done; } while(0)

    if ((sbuf[0] == 'g') && (sbuf[1] == 't') &&
        strchr(specials, sbuf[2])) {
        lexer_return('>', 2);
    } else if ((sbuf[0] == 'l') && (sbuf[1] == 't') &&
               strchr(specials, sbuf[2])) {
        lexer_return('<', 2);
    } else if ((sbuf[0] == 'g') && (sbuf[1] == 'e') &&
               strchr(specials, sbuf[2])) {
        lexer_return(TOK_GE, 2);
    } else if ((sbuf[0] == 'l') && (sbuf[1] == 'e') &&
               strchr(specials, sbuf[2])) {
        lexer_return(TOK_LE, 2);
    } else if ((sbuf[0] == 'n') && (sbuf[1] == 'e') &&
               strchr(specials, sbuf[2])) {
        lexer_return(TOK_NE, 2);
    } else if ((sbuf[0] == 'e') && (sbuf[1] == 'q') &&
               strchr(specials, sbuf[2])) {
        lexer_return('=', 2);
    } else if ((sbuf[0] == 'o') && (sbuf[1] == 'r') &&
               strchr(specials, sbuf[2])) {
        lexer_return('|', 2);
    } else if ((sbuf[0] == 'a') && (sbuf[1] == 'n') &&
               (sbuf[2] == 'd') && strchr(specials, sbuf[3])) {
        lexer_return('&', 3);
    } else if ((sbuf[0] == 'n') && (sbuf[1] == 'o') &&
               (sbuf[2] == 't') && strchr(specials, sbuf[3])) {
        lexer_return('~', 3);
    }

    switch (*sbuf) {

    case '[':
        if (sbuf[1] == '[') {
            lexer_return(TOK_LRANGE, 2);
        } else {
            lexer_return(*sbuf, 1);
        }

    case ']':
        if (sbuf[1] == ']') {
            lexer_return(TOK_RRANGE, 2);
        } else {
            lexer_return(*sbuf, 1);
        }

    case '>':
    case '<':
    {
        /* Workaround, The Frontend makes "<>" into "< >" */
        int j = 1;
        while (isspace(sbuf[j]))
            j++;
        if (((sbuf[j] == '<') || (sbuf[j] == '>')) && (sbuf[0] != sbuf[j])) {
            /* Allow both <> and >< for NE. */
            lexer_return(TOK_NE, j+1);
        } else if (sbuf[1] == '=') {
            lexer_return((sbuf[0] == '>') ? TOK_GE : TOK_LE, 2);
        } else {
            lexer_return(*sbuf, 1);
        }
    }

    case '?':
    case ':':
    case ',':
    case '+':
    case '-':
    case '*':
    case '%':
    case '/':
    case '^':
    case '(':
    case ')':
    case '=':
    case '&':
    case '|':
    case '~':
        lexer_return(*sbuf, 1);

    case '\0':
        lexer_return(*sbuf, 0);

    case '"':
    {
        char *start = ++sbuf;
        while (*sbuf && (*sbuf != '"'))
            sbuf++;
        lvalp->str = copy_substring(start, sbuf);
        if (*sbuf)
            sbuf++;
        lexer_return(TOK_STR, 0);
    }

    default:
    {
        char *s = sbuf;
        double *td = ft_numparse(&s, FALSE);
        if ((!s || *s != ':') && td) {
            sbuf = s;
            lvalp->num = *td;
            lexer_return(TOK_NUM, 0);
        } else {
            int atsign = 0;
            char *start = sbuf;
            /* It is bad how we have to recognise '[' -- sometimes
             * it is part of a word, when it defines a parameter
             * name, and otherwise it isn't.
             * va, ']' too
             */
            for (; *sbuf && !strchr(specials, *sbuf); sbuf++)
                if (*sbuf == '@')
                    atsign = 1;
                else if (!atsign && (*sbuf == '[' || *sbuf == ']'))
                    break;

            lvalp->str = copy_substring(start, sbuf);
            lexer_return(TOK_STR, 0);
        }
    }
    }

done:
    if (ft_parsedb) {
        if (token == TOK_STR)
            fprintf(stderr, "lexer: TOK_STR, \"%s\"\n", lvalp->str);
        else if (token == TOK_NUM)
            fprintf(stderr, "lexer: TOK_NUM, %G\n", lvalp->num);
        else
            fprintf(stderr, "lexer: token %d\n", token);
    }

    *line = sbuf;
    llocp->stop = sbuf;
    return (token);
}
