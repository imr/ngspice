/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Wayne A. Christopher, U. C. Berkeley CAD Group
**********/

/*
 * Control language parser:
 * A simple operator-precedence parser for algebraic expressions.
 * This also handles relational and logical expressions.
 */

#include "ngspice/ngspice.h"
#include "ngspice/bool.h"
#include "ngspice/fteparse.h"
#include "ngspice/fteext.h"
#include "ngspice/sim.h"
#include "numparam/general.h"

#include "evaluate.h"
#include "parse.h"
#include "parse-bison.h"
#include "parse-bison-y.h"


static bool checkvalid(struct pnode *pn);

#ifdef OLD_BISON
extern int PPparse(char **, struct pnode **);
#endif

void db_print_pnode_tree(struct pnode *p, char *print);

struct pnode *ft_getpnames_from_string(const char *sz, bool check)
{
    struct pnode *pn;

    /* The first argument to PPparse is not const char **, but it does not
     * appear to modify the string that is being parsed */
    if (PPparse((char **) &sz, &pn) != 0) {
        return (struct pnode *) NULL;
    }

    /* If validation is requested, do it and return NULL on failure. The
     * structure must also be freed if the check fails since it is not
     * being returned. */
    if (check && !checkvalid(pn)) {
        vec_free_x(pn->pn_value);
        free_pnode(pn);
        return (struct pnode *) NULL;
    }

    return pn;
} /* end of function ft_getpnames_from_string */



struct pnode *
ft_getpnames(const wordlist *wl, bool check)
{
    /* Validate input */
    if (!wl) {
        (void) fprintf(cp_err, "Warning: NULL arithmetic expression\n");
        return (struct pnode *) NULL;
    }

    /* Convert the list to a string, then parse the string */
    const char * const sz = wl_flatten(wl);
    struct pnode * const pn = ft_getpnames_from_string(sz, check);
    txfree((void *) sz);

    return pn; /* Return the parsed result */
} /* end of function ft_getpnames */



static bool is_all_digits(char* tstr)
{
    while (*tstr != '\0') {
        if (!isdigit_c(*tstr))
            return FALSE;
        tstr++;
    }
    return TRUE;
}

static bool has_arith_char(char* tstr)
{
    while (*tstr != '\0') {
        if (is_arith_char(*tstr))
            return TRUE;
        tstr++;
    }
    return FALSE;
}

/* writing, printing or plotting will fail when the node name starts with
   a number or math character, even when enclosed in V() like V(2p). So
   automatically place "" around, like V("2p"). Returns the parse tree. Multiple
   v() may occur in a row. Remove "" again after the tree is set up.
*/
struct pnode* ft_getpnames_quotes(wordlist* wl, bool check)
{
    struct pnode* names = NULL, * tmpnode = NULL;
    char* sz = wl_flatten(wl);
    if ((strstr(sz, "v(") || strstr(sz, "V(") || strstr(sz, "i(") || strstr(sz, "I(")) && !cp_getvar("noquotesinoutput", CP_BOOL, NULL, 0))
    {
        char* tmpstr;
        char* nsz = tmpstr = stripWhiteSpacesInsideParens(sz);
        DS_CREATE(ds1, 100); /* the new name string*/
        /* put double quotes around tokens which start with number chars or include a math char */
        while (*tmpstr != '\0') {
            /*check if we have v(something) at the beginning, after arithchar, after space,
              or after dot. Skip V(" because it is already quoted. */
            if ((tmpstr[0] == 'v' || tmpstr[0] == 'V') && tmpstr[1] == '('  && tmpstr[2] != '\"' &&
                    (nsz == tmpstr || isspace_c(tmpstr[-1]) || is_arith_char(tmpstr[-1]) || tmpstr[-1] == '.')) {
                char* tmpstr2, * partoken2 = NULL;
                tmpstr += 2;
                /* get the complete zzz of v(zzz) */
                char* tpartoken = tmpstr2 = gettok_char(&tmpstr, ')', FALSE, FALSE);
                /* check if this is v(zzz) or v(xx,yy) */
                char* partoken1 = gettok_char(&tpartoken, ',', FALSE, FALSE);
                sadd(&ds1, "v(");
                if (partoken1) {
                    /* we have a xx and yy */
                    partoken2 = copy(tpartoken + 1);
                    bool hac1 = has_arith_char(partoken1);
                    bool hac2 = has_arith_char(partoken2);
                    if (is_all_digits(partoken1)) {
                        sadd(&ds1, partoken1);
                    }
                    else if (isdigit_c(*partoken1) || hac1) {
                        cadd(&ds1, '\"');
                        sadd(&ds1, partoken1);
                        cadd(&ds1, '\"');
                    }
                    else
                        sadd(&ds1, partoken1);
                    cadd(&ds1, ',');
                    if (is_all_digits(partoken2)) {
                        sadd(&ds1, partoken2);
                    }
                    else if (isdigit_c(*partoken2) || hac2) {
                        cadd(&ds1, '\"');
                        sadd(&ds1, partoken2);
                        cadd(&ds1, '\"');
                    }
                    else
                        sadd(&ds1, partoken2);
                }
                else {
                    bool hac = has_arith_char(tmpstr2);
                    if (is_all_digits(tmpstr2)) {
                        sadd(&ds1, tmpstr2);
                    }
                    else if (isdigit_c(*tmpstr2) || hac) {
                        cadd(&ds1, '\"');
                        sadd(&ds1, tmpstr2);
                        cadd(&ds1, '\"');
                    }
                    else
                        sadd(&ds1, tmpstr2);
                }

                tfree(tmpstr2);
                tfree(partoken1);
                tfree(partoken2);
            }
            else if ((tmpstr[0] == 'i' || tmpstr[0] == 'I') && tmpstr[1] == '(' && tmpstr[2] != '\"' &&
                (nsz == tmpstr || isspace_c(tmpstr[-1]) || is_arith_char(tmpstr[-1]) || tmpstr[-1] == '.')) {
                char* tmpstr2, *tmpstr3;
                tmpstr3 = tmpstr;
                tmpstr += 2;
                /* get the complete zzz of i(zzz) */
                tmpstr2 = gettok_char(&tmpstr, ')', FALSE, FALSE);
                /* missing final ) ?*/
                if (!tmpstr2) {
                    fprintf(stderr, "Error: closing ) is missing in %s,\n    ignored\n", tmpstr3);
                    tmpstr = ++tmpstr3;
                    continue;
                }
                /* check if this is i(zzz) or v(xx,yy) */
                sadd(&ds1, "i(");

                    bool hac = has_arith_char(tmpstr2);
                    if (is_all_digits(tmpstr2)) {
                        sadd(&ds1, tmpstr2);
                    }
                    else if (isdigit_c(*tmpstr2) || hac) {
                        cadd(&ds1, '\"');
                        sadd(&ds1, tmpstr2);
                        cadd(&ds1, '\"');
                    }
                    else
                        sadd(&ds1, tmpstr2);

                tfree(tmpstr2);
             }
            cadd(&ds1, *tmpstr);
            tmpstr++;
        }

        char* newline = ds_get_buf(&ds1);
        names = ft_getpnames_from_string(newline, check);
        ds_free(&ds1);
        tfree(nsz);
        /* restore the old node name after parsing */
        for (tmpnode = names; tmpnode; tmpnode = tmpnode->pn_next) {
            if (strstr(tmpnode->pn_name, "v(\"") || strstr(tmpnode->pn_name, "i(\"")) {
                char newstr[100];
                char* tmp = tmpnode->pn_name;
                int ii = 0;
                // copy to newstr without double quotes
                while (*tmp && ii < 99) {
                    if (*(tmp) == '\"') {
                        tmp++;
                        continue;
                    }
                    newstr[ii] = *(tmp++);
                    ii++;
                }
                newstr[ii] = '\0';
                tfree(tmpnode->pn_name);
                tmpnode->pn_name = copy(newstr);
            }
        }
    }
    else {
        names = ft_getpnames_from_string(sz, check);
    }
    tfree(sz);
    return names;
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
                            "Warning from checkvalid: %s: no matching vectors.\n",
                            pn->pn_value->v_name);
                else
                    fprintf(cp_err,
                            "Warning from checkvalid: vector %s is not available or has zero length.\n",
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
 *
 * When these functions are called (apply_func_funcall() evaluate.c),
 * the actual argument list is longer than declared here.  Only the functions
 * with casts use the extra arguments.  The double casts prevent warnings
 * with some gcc versions.
 */

typedef void* cx_function_t(void*, short int, int, int*, short int*);

struct func ft_funcs[] = {
    { "mag",         cx_mag },
    { "magnitude",   cx_mag },
    { "cph",         cx_cph },  /* SJdV */
    { "cphase",      cx_cph },  /* SJdV Continious phase*/
    { "unwrap",      cx_unwrap },
    { "ph",          cx_ph },
    { "phase",       cx_ph },
    { "j",           cx_j },
    { "real",        cx_real },
    { "re",          cx_real },
    { "imag",        cx_imag },
    { "im",          cx_imag },
    { "conj",        cx_conj },
    { "db",          cx_db },
    { "log",         cx_log },
    { "log10",       cx_log10 },
    { "ln",          cx_log },
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
    { "atanh",       cx_atanh },
    { "sortorder",   cx_sortorder },
    { "norm",        cx_norm },
    { "rnd",         cx_rnd },
    { "sunif",       cx_sunif },
    { "poisson",     cx_poisson },
    { "exponential", cx_exponential },
    { "sgauss",      cx_sgauss },
    { "pos",         cx_pos },
    { "nint",        cx_nint },
    { "floor",       cx_floor },
    { "ceil",        cx_ceil },
    { "mean",        cx_mean },
    { "stddev",      cx_stddev },
    { "avg",         cx_avg }, /* A.Roldan 03/06/05 incremental average new function */
    { "group_delay", (cx_function_t*)(void *) cx_group_delay }, /* A.Roldan 10/06/05 group delay new function */
    { "vector",      cx_vector },
    { "cvector",     cx_cvector },
    { "unitvec",     cx_unitvec },
    { "length",      cx_length },
    { "vecmin",      cx_min },
    { "minimum",     cx_min },
    { "vecmax",      cx_max },
    { "maximum",     cx_max },
    { "vecd",        cx_d },
    { "interpolate", (cx_function_t*)(void *) cx_interpolate },
    { "deriv",       (cx_function_t*)(void *) cx_deriv },
    { "integ",       (cx_function_t*)(void *) cx_integ },
    { "fft",         (cx_function_t*)(void *) cx_fft },
    { "ifft",        (cx_function_t*)(void *) cx_ifft },
    { "v",           NULL },
    { NULL,          NULL }
};

struct func func_uminus = { "minus", cx_uminus };

struct func func_not = { "not", cx_not };


/* Binary operator node. */
struct pnode *PP_mkbnode(int opnum, struct pnode *arg1, struct pnode *arg2)
{
    struct op *o;
    struct pnode *p;

    for (o = &ops[0]; o->op_name; o++) {
        if (o->op_num == opnum) {
            break;
        }
    }

    if (!o->op_name) {
        fprintf(cp_err, "PP_mkbnode: Internal Error: no such op num %d\n",
                opnum);
    }

    p = alloc_pnode();

    p->pn_op = o;

    p->pn_left = arg1;
    if (p->pn_left) {
        p->pn_left->pn_use++;
    }

    p->pn_right = arg2;
    if (p->pn_right) {
        p->pn_right->pn_use++;
    }

    return p;
} /* end of function PP_mkbnode */



/* Unary operator node. */
struct pnode *PP_mkunode(int op, struct pnode *arg)
{
    struct pnode *p;
    struct op *o;

    p = alloc_pnode();

    for (o = uops; o->op_name; o++) {
        if (o->op_num == op) {
            break;
        }
    }

    if (!o->op_name) {
        fprintf(cp_err, "PP_mkunode: Internal Error: no such op num %d\n",
                op);
    }

    p->pn_op = o;

    p->pn_left = arg;
    if (p->pn_left) {
        p->pn_left->pn_use++;
    }

    return p;
} /* end of function PP_mkunode */



/* Function node. We have to worry about a lot of things here. Something
 * like f(a) could be three things -- a call to a standard function, which
 * is easiest to deal with, a variable name, in which case we do the
 * kludge with 0-length lists, or it could be a user-defined function,
 * in which case we have to figure out which one it is, substitute for
 * the arguments, and then return a copy of the expression that it was
 * defined to be.
 */
struct pnode *PP_mkfnode(const char *func, struct pnode *arg)
{
    struct func *f;
    struct pnode *p, *q;
    struct dvec *d;
    char buf[BSIZE_SP];

    (void) strcpy(buf, func);
    strtolower(buf);  /* Make sure the case is ok. */

    for (f = &ft_funcs[0]; f->fu_name; f++) {
        if (eq(f->fu_name, buf)) {
            break;
        }
    }

    if (f->fu_name == NULL) { /* not found yet */
        /* Give the user-defined functions a try. */
        q = ft_substdef(func, arg);
        if (q) { /* found */
            /* remove only the old comma operator pnode, no longer used */
            if (arg->pn_op && arg->pn_op->op_num == PT_OP_COMMA) {
                free_pnode(arg);
            }
            return q;
        }
    }

    if ((f->fu_name == NULL) && arg->pn_value) {
        /* Kludge -- maybe it is really a variable name. */
        (void) sprintf(buf, "%s(%s)", func, arg->pn_value->v_name);
        free_pnode(arg);
        d = vec_get(buf);
        if (d == NULL) {
            /* Well, too bad. */
            fprintf(cp_err, "\nError: no such function as %s,\n",
                    func);
            fprintf(cp_err, "    or %s is not available.\n",
                    buf);
            return (struct pnode *) NULL;
        }
        /* (void) strcpy(buf, d->v_name); XXX */
        return PP_mksnode(buf);
    }
    else if (f->fu_name == NULL) {
        fprintf(cp_err, "Error: no function as %s with that arity.\n",
                func);
        free_pnode(arg);
        return (struct pnode *) NULL;
    }

    if (!f->fu_func && arg->pn_op && arg->pn_op->op_num == PT_OP_COMMA) {
        p = PP_mkbnode(PT_OP_MINUS, PP_mkfnode(func, arg->pn_left),
                    PP_mkfnode(func, arg->pn_right));
        free_pnode(arg);
        return p;
    }

    p = alloc_pnode();

    p->pn_func = f;

    p->pn_left = arg;
    if (p->pn_left) {
        p->pn_left->pn_use++;
    }

    return p;
} /* end of function PP_mkfnode */



/* Number node. */
struct pnode *PP_mknnode(double number)
{
    struct pnode *p;
    struct dvec *v;

    /* We don't use printnum because it screws up PP_mkfnode above. We have
     * to be careful to deal properly with node numbers that are quite
     * large...
     */
    v = dvec_alloc(number <= INT_MAX
                   ? tprintf("%d", (int) number)
                   : tprintf("%G", number),
                   SV_NOTYPE,
                   VF_REAL,
                   1, NULL);

    v->v_realdata[0] = number;

    vec_new(v);

    p = alloc_pnode();
    p->pn_value = v;
    return (p);
} /* end of function PP_mknnode */



/* String node. */
struct pnode *PP_mksnode(const char *string)
{
    struct dvec *v, *nv, *vs, *newv = NULL, *end = NULL;
    struct pnode *p;

    p = alloc_pnode();
    v = vec_get(string);
    if (v == NULL) {
        nv = dvec_alloc(copy(string),
                        SV_NOTYPE,
                        0,
                        0, NULL);
        p->pn_value = nv;
        return p;
    }

    /* It's not obvious that we should be doing this, but... */
    for (vs = v; vs; vs = vs->v_link2) {
        nv = vec_copy(vs);
        vec_new(nv);
        if (end) {
            end->v_link2 = nv;
        }
        else {
            newv = end = nv;
        }
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
    return p;
} /* end of function PP_mksnode */



struct pnode *alloc_pnode(void)
{
    struct pnode *pn = TMALLOC(struct pnode, 1);

    pn->pn_use = 0;
    pn->pn_name = NULL;

    // fixme, thats actually a union ...
    pn->pn_value = NULL;
    pn->pn_func = NULL;
    pn->pn_op = NULL;

    pn->pn_left = NULL;
    pn->pn_right = NULL;
    pn->pn_next = NULL;

    return pn;
} /* end of function alloc_pnode */



/* Don't call this directly, always use the free_pnode() macro.
   The linked pnodes do not necessarily form a perfect tree as some nodes get
   reused.  Hence, in this recursive walk through the 'tree', we only free
   nodes that have their pn_use value at zero. Nodes that have pn_use values
   above zero have the link severed and their pn_use value decremented.
   In addition, we don't walk past nodes with pn_use values avoid zero, just
   in case we have a circular reference (This probably does not happen in
   practice, but it does no harm playing safe.) */
void free_pnode_x(struct pnode *t)
{
    if (!t) {
        return;
    }

    /* Don't walk past nodes used elsewhere. We decrement the pn_use value here,
       but the link gets severed by the action of the free_pnode() macro */
    if (t->pn_use > 1) {
        t->pn_use--;
    }
    else {
        /* pn_use is now 1, so its safe to free the pnode */
        free_pnode(t->pn_left);
        free_pnode(t->pn_right);
        free_pnode(t->pn_next);
        tfree(t->pn_name); /* va: it is a copy() of original string, can be free'd */
        if (t->pn_use == 1 && t->pn_value && !(t->pn_value->v_flags & VF_PERMANENT)) {
            vec_free(t->pn_value); /* patch by Stefan Jones */
        }
        txfree(t);
    }
} /* end of function free_pnode_x */



static void db_print_func(FILE *fdst, struct func *f)
{
    if (!f) {
        fprintf(fdst, "nil");
        return;
    }

    fprintf(fdst, "(func :fu_name %s :fu_func %p)", f->fu_name, f->fu_func);
} /* end of function db_print_func */



static void db_print_op(FILE *fdst, struct op *op)
{
    if (!op) {
        fprintf(fdst, "nil");
        return;
    }

    fprintf(fdst, "(op :op_num %d :op_name %s :op_arity %d :op_func %p)",
            op->op_num, op->op_name, op->op_arity, op->op_func.anonymous);
} /* end of function db_print_op */



static void db_print_dvec(FILE *fdst, struct dvec *d)
{
    if (!d) {
        fprintf(fdst, "nil");
        return;
    }

    fprintf(fdst, "(dvec :v_name %s :v_type %d :v_flags %d :v_length %d ...)",
            d->v_name, d->v_type, d->v_flags, d->v_length);
} /* end of function db_print_dvec */



static void db_print_pnode(FILE *fdst, struct pnode *p)
{
    if (!p) {
        fprintf(fdst, "nil\n");
        return;
    }

    if (!p->pn_name && p->pn_value && !p->pn_func && !p->pn_op &&
        !p->pn_left && !p->pn_right && !p->pn_next) {
        fprintf(fdst, "(pnode-value :pn_use %d", p->pn_use);
        fprintf(fdst, " :pn_value "); db_print_dvec(fdst, p->pn_value);
        fprintf(fdst, ")\n");
        return;
    }

    if (!p->pn_name && !p->pn_value && p->pn_func && !p->pn_op &&
        !p->pn_right && !p->pn_next) {
        fprintf(fdst, "(pnode-func :pn_use %d", p->pn_use);
        fprintf(fdst, "\n :pn_func "); db_print_func(fdst, p->pn_func);
        fprintf(fdst, "\n :pn_left "); db_print_pnode(fdst, p->pn_left);
        fprintf(fdst, ")\n");
        return;
    }

    if (!p->pn_name && !p->pn_value && !p->pn_func && p->pn_op &&
        !p->pn_next) {
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
} /* end of function db_print_pnode */



void db_print_pnode_tree(struct pnode *p, char *print)
{
#if 1
    NG_IGNORE(print);
    db_print_pnode(stdout, p);
#else
    char *buf;
    size_t  buf_size;
    FILE *db_stream = open_memstream(&buf, &buf_size);
    db_print_pnode(db_stream, p);
    fclose(db_stream);
    if (print)
        printf("%s:%d: %s {%s}\n%s\n", __FILE__, __LINE__, __func__, print, buf);
    tfree(buf);
#endif
} /* end of function db_print_pnode_tree */



int PPlex(YYSTYPE *lvalp, struct PPltype *llocp, char **line)
{
    static char *specials = " \t%()-^+*,/|&<>~=";
    char  *sbuf = *line;
    int token;

    while ((*sbuf == ' ') || (*sbuf == '\t')) {
        sbuf++;
    }

    llocp->start = sbuf;

#define lexer_return(token_, length)                            \
    do { token = token_; sbuf += length; goto done; } while(0)

    if ((sbuf[0] == 'g') && (sbuf[1] == 't') &&
        strchr(specials, sbuf[2])) {
        lexer_return('>', 2);
    }
    else if ((sbuf[0] == 'l') && (sbuf[1] == 't') &&
               strchr(specials, sbuf[2])) {
        lexer_return('<', 2);
    }
    else if ((sbuf[0] == 'g') && (sbuf[1] == 'e') &&
               strchr(specials, sbuf[2])) {
        lexer_return(TOK_GE, 2);
    }
    else if ((sbuf[0] == 'l') && (sbuf[1] == 'e') &&
               strchr(specials, sbuf[2])) {
        lexer_return(TOK_LE, 2);
    }
    else if ((sbuf[0] == 'n') && (sbuf[1] == 'e') &&
               strchr(specials, sbuf[2])) {
        lexer_return(TOK_NE, 2);
    }
    else if ((sbuf[0] == 'e') && (sbuf[1] == 'q') &&
               strchr(specials, sbuf[2])) {
        lexer_return('=', 2);
    }
    else if ((sbuf[0] == 'o') && (sbuf[1] == 'r') &&
               strchr(specials, sbuf[2])) {
        lexer_return('|', 2);
    }
    else if ((sbuf[0] == 'a') && (sbuf[1] == 'n') &&
               (sbuf[2] == 'd') && strchr(specials, sbuf[3])) {
        lexer_return('&', 3);
    }
    else if ((sbuf[0] == 'n') && (sbuf[1] == 'o') &&
               (sbuf[2] == 't') && strchr(specials, sbuf[3])) {
        lexer_return('~', 3);
    }

    switch (*sbuf) {

    case '[':
    case ']':
        lexer_return(*sbuf, 1);

    case '>':
    case '<': {
        /* Workaround, The Frontend makes "<>" into "< >" */
        size_t j = 1;
        while (isspace_c(sbuf[j]))
            j++;
        if (((sbuf[j] == '<') || (sbuf[j] == '>')) && (sbuf[0] != sbuf[j])) {
            /* Allow both <> and >< for NE. */
            lexer_return(TOK_NE, j + 1);
        }
        else if (sbuf[1] == '=') {
            lexer_return((sbuf[0] == '>') ? TOK_GE : TOK_LE, 2);
        }
        else {
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

    case '"': {
        char *start = ++sbuf;
        while (*sbuf && (*sbuf != '"'))
            sbuf++;
        lvalp->str = copy_substring(start, sbuf);
        if (*sbuf) {
            sbuf++;
        }
        lexer_return(TOK_STR, 0);
    }

    default: {
        char *s = sbuf;
        double val;

        if (ft_numparse(&s, FALSE, &val) >= 0 &&
                (!s || *s != ':')) {
            sbuf = s;
            lvalp->num = val;
            lexer_return(TOK_NUM, 0);
        }
        else {
            int atsign = 0;
            char *start = sbuf;
            /* It is bad how we have to recognise '[' -- sometimes
             * it is part of a word, when it defines a parameter
             * name, and otherwise it isn't.
             *
             * what is valid here ?
             *   foo  dc1.foo  dc1.@m1[vth]
             *   vthing#branch
             *   i(vthing)
             */
            for (; *sbuf && !strchr(specials, *sbuf); sbuf++) {
                if (*sbuf == '@') {
                    atsign = 1;
                }
                else if (!atsign && *sbuf == '[') {
                    break;
                }
                else if (*sbuf == ']') {
                    if (atsign) {
                        sbuf++;
                    }
                    break;
                } else if ((sbuf == start || sbuf[-1] == '.') &&
                           prefix("i(v", sbuf)) {
                    /* Special case for current through voltage source:
                     * keep the identifier i(vss) as a single token,
                     * even as dc1.i(vss).
                     */

                    if (get_r_paren(&sbuf) == 1) {
                        fprintf(stderr,
                                "Error: missing ')' in token\n    %s\n",
                                start);
                        break;
                    }
                    sbuf--; // Point at ')', last accepted char.
                }
            }
            lvalp->str = copy_substring(start, sbuf);
            lexer_return(TOK_STR, 0);
        }
    }
    } /* end of switch over characters */

done:
    if (ft_parsedb) {
        if (token == TOK_STR) {
            fprintf(stderr, "lexer: TOK_STR, \"%s\"\n", lvalp->str);
        }
        else if (token == TOK_NUM) {
            fprintf(stderr, "lexer: TOK_NUM, %G\n", lvalp->num);
        }
        else {
            fprintf(stderr, "lexer: token %d\n", token);
        }
    }

    *line = sbuf;
    llocp->stop = sbuf;
    return token;
} /* end of function PPlex */



