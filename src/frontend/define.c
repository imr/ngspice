/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Wayne A. Christopher, U. C. Berkeley CAD Group
**********/

/*
 * User-defined functions. The user defines the function with
 *  define func(arg1, arg2, arg3) <expression involving args...>
 * Then when he types "func(1, 2, 3)", the commas are interpreted as
 * binary operations of the lowest priority by the parser, and ft_substdef()
 * below is given a chance to fill things in and return what the parse tree
 * would have been had the entire thing been typed.
 * Note that we have to take some care to distinguish between functions
 * with the same name and different arities.
 */

#include "ngspice/ngspice.h"
#include "ngspice/cpdefs.h"
#include "ngspice/ftedefs.h"
#include "ngspice/dvec.h"
#include "ngspice/fteparse.h"
#include "define.h"

#include "completion.h"


static void savetree(struct pnode *pn);
static void prdefs(char *name);
static void prtree(struct udfunc *ud, FILE *fp);
static void prtree1(struct pnode *pn, FILE *fp);
static struct pnode *trcopy(struct pnode *tree, char *arg_names, struct pnode *args);
static struct pnode *ntharg(int num, struct pnode *args);
static int numargs(struct pnode *args);

static struct udfunc *udfuncs = NULL;


/* Set up a function definition. */

void
com_define(wordlist *wlist)
{
    int arity = 0, i;
    char buf[BSIZE_SP], tbuf[BSIZE_SP], *s, *t, *b;
    wordlist *wl;
    struct pnode *names;
    struct udfunc *udf;

    /* If there's nothing then print all the definitions. */
    if (wlist == NULL) {
        prdefs(NULL);
        return;
    }

    /* Accumulate the function head in the buffer, w/out spaces. A
     * useful thing here would be to check to make sure that there
     * are no formal parameters here called "list". But you have
     * to try really hard to break this here.
     */
    buf[0] = '\0';

    for (wl = wlist; wl && (strchr(wl->wl_word, ')') == NULL);
         wl = wl->wl_next)
        (void) strcat(buf, wl->wl_word);

    if (wl) {
        t = strchr(buf, '\0');
        for (s = wl->wl_word; *s && (*s != ')');)
            *t++ = *s++;
        *t++ = ')';
        *t = '\0';
        if (*++s)
            wl->wl_word = copy(s);
        else
            wl = wl->wl_next;
    }

    /* If that's all, then print the definition. */
    if (wl == NULL) {
        s = strchr(buf, '(');
        if (s)
            *s = '\0';
        prdefs(buf);
        return;
    }

    /* Now check to see if this is a valid name for a function (i.e,
     * there isn't a predefined function of the same name).
     */
    (void) strcpy(tbuf, buf);

    for (b = tbuf; *b; b++)
        if (isspace_c(*b) || (*b == '(')) {
            *b = '\0';
            break;
        }

    for (i = 0; ft_funcs[i].fu_name; i++)
        if (eq(ft_funcs[i].fu_name, tbuf)) {
            fprintf(cp_err, "Error: %s is a predefined function.\n",
                    tbuf);
            return;
        }

    /* Parse the rest of it. We can't know if there are the right
     * number of undefined variables in the expression.
     */
    if ((names = ft_getpnames(wl, FALSE)) == NULL)
        return;

    /* This is a pain -- when things are garbage-collected, any
     * vectors that may have been mentioned here will be thrown
     * away. So go down the tree and save any vectors that aren't
     * formal parameters.
     */
    savetree(names);

    /* Format the name properly and add to the list. */
    b = copy(buf);
    for (s = b; *s; s++) {
        if (*s == '(') {
            *s = '\0';
            if (s[1] != ')')
                arity++;    /* It will have been 0. */
        } else if (*s == ')') {
            *s = '\0';
        } else if (*s == ',') {
            *s = '\0';
            arity++;
        }
    }

    for (udf = udfuncs; udf; udf = udf->ud_next)
        if (prefix(b, udf->ud_name) && (arity == udf->ud_arity))
            break;

    if (udf == NULL) {
        udf = TMALLOC(struct udfunc, 1);
        udf->ud_next = udfuncs;
        udfuncs = udf;
    }

    udf->ud_text = names;
    udf->ud_name = b;
    udf->ud_arity = arity;

    cp_addkword(CT_UDFUNCS, b);
}


/* Kludge. */

static void
savetree(struct pnode *pn)
{
    struct dvec *d;

    if (pn->pn_value) {
        /* We specifically don't add this to the plot list
         * so it won't get gc'ed.
         */
        d = pn->pn_value;
        if ((d->v_length != 0) || eq(d->v_name, "list")) {
            pn->pn_value = dvec_alloc(copy(d->v_name),
                                      d->v_type,
                                      d->v_flags,
                                      d->v_length, NULL);

            /* this dvec isn't member of any plot */

            if (isreal(d)) {
                memcpy(pn->pn_value->v_realdata,
                      d->v_realdata,
                      sizeof(double) * (size_t) d->v_length);
            } else {
                memcpy(pn->pn_value->v_compdata,
                      d->v_compdata,
                      sizeof(ngcomplex_t) * (size_t) d->v_length);
            }
        }
    } else if (pn->pn_op) {
        savetree(pn->pn_left);
        if (pn->pn_op->op_arity == 2)
            savetree(pn->pn_right);
    } else if (pn->pn_func) {
        savetree(pn->pn_left);
    }
}


/* A bunch of junk to print out nodes. */

static void
prdefs(char *name)
{
    struct udfunc *udf;

    if (name && *name) {    /* You never know what people will do */
        for (udf = udfuncs; udf; udf = udf->ud_next)
            if (eq(name, udf->ud_name))
                prtree(udf, cp_out);
    } else {
        for (udf = udfuncs; udf; udf = udf->ud_next)
            prtree(udf, cp_out);
    }
}


/* Print out one definition. */

static void
prtree(struct udfunc *ud, FILE *fp)
{
    const char *s = ud->ud_name;

    /* print the function name */
    fprintf(fp, "%s (", s);
    s = strchr(s, '\0') + 1;

    /* print the formal args */
    while (*s) {
        fputs(s, fp);
        s = strchr(s, '\0') + 1;
        if (*s)
            fputs(", ", fp);
    }
    fputs(") = ", fp);

    /* print the function body */
    prtree1(ud->ud_text, fp);
    putc('\n', fp);
}


static void
prtree1(struct pnode *pn, FILE *fp)
{
    if (pn->pn_value) {
        fputs(pn->pn_value->v_name, fp);
    } else if (pn->pn_func) {
        fprintf(fp, "%s (", pn->pn_func->fu_name);
        prtree1(pn->pn_left, fp);
        fputs(")", fp);
    } else if (pn->pn_op && (pn->pn_op->op_arity == 2)) {
        fputs("(", fp);
        prtree1(pn->pn_left, fp);
        fprintf(fp, ")%s(", pn->pn_op->op_name);
        prtree1(pn->pn_right, fp);
        fputs(")", fp);
    } else if (pn->pn_op && (pn->pn_op->op_arity == 1)) {
        fprintf(fp, "%s(", pn->pn_op->op_name);
        prtree1(pn->pn_left, fp);
        fputs(")", fp);
    } else {
        fputs("<something strange>", fp);
    }
}


struct pnode *
ft_substdef(const char *name, struct pnode *args)
{
    struct udfunc *udf, *wrong_udf = NULL;
    char *arg_names;

    int arity = numargs(args);

    for (udf = udfuncs; udf; udf = udf->ud_next)
        if (eq(name, udf->ud_name)) {
            if (arity == udf->ud_arity)
                break;
            wrong_udf = udf;
        }

    if (udf == NULL) {
        if (wrong_udf)
            fprintf(cp_err,
                    "Warning: the user-defined function %s has %d args\n",
                    name, wrong_udf->ud_arity);
        return NULL;
    }

    arg_names = strchr(udf->ud_name, '\0') + 1;

    /* Now we have to traverse the tree and copy it over,
     * substituting args.
     */
    return trcopy(udf->ud_text, arg_names, args);
}


/* Copy the tree and replace formal args with the right stuff. The way
 * we know that something might be a formal arg is when it is a dvec
 * with length 0 and a name that isn't "list". I hope nobody calls their
 * formal parameters "list".
 */

static struct pnode *
trcopy(struct pnode *tree, char *arg_names, struct pnode *args)
{
    if (tree->pn_value) {

        struct dvec *d = tree->pn_value;

        if ((d->v_length == 0) && strcmp(d->v_name, "list")) {

            /* Yep, it's a formal parameter. Substitute for it.
             * IMPORTANT: we never free parse trees, so we
             * needn't worry that they aren't trees here.
             */

            char *s = arg_names;
            int i;

            for (i = 1; *s; i++) {
                if (eq(s, d->v_name))
                    return ntharg(i, args);
                s = strchr(s, '\0') + 1;
            }

            return tree;
        }

        return tree;
    }

    if (tree->pn_func) {

        struct pnode *pn = alloc_pnode();

        /* pn_func are pointers to a global constant struct */
        pn->pn_func = tree->pn_func;

        pn->pn_left = trcopy(tree->pn_left, arg_names, args);
        pn->pn_left->pn_use++;

        return pn;
    }

    if (tree->pn_op) {

        struct pnode *pn = alloc_pnode();

        /* pn_op are pointers to a global constant struct */
        pn->pn_op = tree->pn_op;

        pn->pn_left = trcopy(tree->pn_left, arg_names, args);
        pn->pn_left->pn_use++;

        if (pn->pn_op->op_arity == 2) {
            pn->pn_right = trcopy(tree->pn_right, arg_names, args);
            pn->pn_right->pn_use++;
        }

        return pn;
    }

    fprintf(cp_err, "trcopy: Internal Error: bad parse node\n");
    return NULL;
}


/* Find the n'th arg in the arglist, returning NULL if there isn't one.
 * Since comma has such a low priority and associates to the right,
 * we can just follow the right branch of the tree num times.
 * Note that we start at 1 when numbering the args.
 */

static struct pnode *
ntharg(int num, struct pnode *args)
{
    for (; args; args = args->pn_right, --num) {
        if (num <= 1) {
            if (args->pn_op && (args->pn_op->op_num == PT_OP_COMMA))
                return args->pn_left;
            return args;
        }
        if (!(args->pn_op && (args->pn_op->op_num == PT_OP_COMMA)))
            return NULL;
    }

    return NULL;
}


static int
numargs(struct pnode *args)
{
    int arity;

    if (!args)
        return 0;

    for (arity = 1; args; args = args->pn_right, arity++)
        if (!(args->pn_op && (args->pn_op->op_num == PT_OP_COMMA)))
            return arity;

    // note: a trailing NULL pn_right will be counted too
    return arity;
}


void
com_undefine(wordlist *wlist)
{
    struct udfunc *udf;

    if (!wlist)
        return;

    if (*wlist->wl_word == '*') {
        for (udf = udfuncs; udf;) {
            struct udfunc *next = udf->ud_next;
            cp_remkword(CT_UDFUNCS, udf->ud_name);
            free_pnode(udf->ud_text);
            tfree(udf->ud_name);
            tfree(udf);
            udf = next;
        }
        udfuncs = NULL;
        return;
    }

    for (; wlist; wlist = wlist->wl_next) {
        struct udfunc *prev_udf = NULL;
        for (udf = udfuncs; udf;) {
            struct udfunc *next = udf->ud_next;
            if (eq(wlist->wl_word, udf->ud_name)) {
                if (prev_udf)
                    prev_udf->ud_next = udf->ud_next;
                else
                    udfuncs = udf->ud_next;
                cp_remkword(CT_UDFUNCS, wlist->wl_word);
                free_pnode(udf->ud_text);
                tfree(udf->ud_name);
                tfree(udf);
            } else {
                prev_udf = udf;
            }
            udf = next;
        }
    }
}


/*
 * This is only here so I can "call" it from gdb/dbx
 */

void
ft_pnode(struct pnode *pn)
{
    prtree1(pn, cp_err);
}
