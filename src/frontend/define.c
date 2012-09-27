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
static void prtree(struct udfunc *ud);
static void prtree1(struct pnode *pn, FILE *fp);
static struct pnode *trcopy(struct pnode *tree, char *args, struct pnode *nn);
static struct pnode *ntharg(int num, struct pnode *args);

static struct udfunc *udfuncs = NULL;


/* Set up a function definition. */

void
com_define(wordlist *wlist)
{
    int arity = 0, i;
    char buf[BSIZE_SP], tbuf[BSIZE_SP], *s, *t, *b;
    wordlist *wl;
    struct pnode *pn;
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

    for (wl = wlist; wl && (strchr(wl->wl_word, /* ( */ ')') == NULL);
         wl = wl->wl_next)
        (void) strcat(buf, wl->wl_word);

    if (wl) {
        for (t = buf; *t; t++)
            ;
        for (s = wl->wl_word; *s && (*s != /* ( */ ')'); s++, t++)
            *t = *s;
        *t++ = /* ( */ ')';
        *t = '\0';
        if (*++s)
            wl->wl_word = copy(s);
        else
            wl = wl->wl_next;
    }

    /* If that's all, then print the definition. */
    if (wl == NULL) {
        prdefs(buf);
        return;
    }

    /* Now check to see if this is a valid name for a function (i.e,
     * there isn't a predefined function of the same name).
     */
    (void) strcpy(tbuf, buf);

    for (b = tbuf; *b; b++)
        if (isspace(*b) || (*b == '(' /* ) */)) {
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
    if ((pn = ft_getpnames(wl, FALSE)) == NULL)
        return;

    /* This is a pain -- when things are garbage-collected, any
     * vectors that may have been mentioned here will be thrown
     * away. So go down the tree and save any vectors that aren't
     * formal parameters.
     */
    savetree(pn);

    /* Format the name properly and add to the list. */
    b = copy(buf);
    for (s = b; *s; s++) {
        if (*s == '(') { /*)*/
            *s = '\0';
            if (s[1] != /*(*/ ')')
                arity++;    /* It will have been 0. */
        } else if (*s == /*(*/ ')') {
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
        udf = alloc(struct udfunc);
        if (udfuncs == NULL) {
            udfuncs = udf;
            udf->ud_next = NULL;
        } else {
            udf->ud_next = udfuncs;
            udfuncs = udf;
        }
    }

    udf->ud_text = pn;
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
            pn->pn_value = alloc(struct dvec);
            ZERO(pn->pn_value, struct dvec);
            pn->pn_value->v_name = copy(d->v_name);
            pn->pn_value->v_length = d->v_length;
            pn->pn_value->v_type = d->v_type;
            pn->pn_value->v_flags = d->v_flags;
            pn->pn_value->v_plot = NULL; /* this dvec isn't member of any plot */
            if (isreal(d)) {
                pn->pn_value->v_realdata = TMALLOC(double, d->v_length);
                bcopy(d->v_realdata,
                      pn->pn_value->v_realdata,
                      sizeof(double) * (size_t) d->v_length);
            } else {
                pn->pn_value->v_compdata = TMALLOC(ngcomplex_t, d->v_length);
                bcopy(d->v_compdata,
                      pn->pn_value->v_compdata,
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
    char *s;

    if (name) {
        s = strchr(name, '(' /* ) */);
        if (s)
            *s = '\0';
    }

    if (name && *name) {    /* You never know what people will do */
        for (udf = udfuncs; udf; udf = udf->ud_next)
            if (eq(name, udf->ud_name))
                prtree(udf);
    } else {
        for (udf = udfuncs; udf; udf = udf->ud_next)
            prtree(udf);
    }
}


/* Print out one definition. */

static void
prtree(struct udfunc *ud)
{
    char *s, buf[BSIZE_SP];

    /* Print the head. */
    buf[0] = '\0';
    (void) strcat(buf, ud->ud_name);
    for (s = ud->ud_name; *s; s++)
        ;
    (void) strcat(buf, " (");
    s++;
    while (*s) {
        (void) strcat(buf, s);
        while (*s)
            s++;
        if (s[1])
            (void) strcat(buf, ", ");
        s++;
    }
    (void) strcat(buf, ") = ");
    fputs(buf, cp_out);
    prtree1(ud->ud_text, cp_out);
    (void) putc('\n', cp_out);
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
    struct udfunc *udf;
    struct pnode *tp;
    char *s;
    int arity = 0, rarity = 0;
    bool found = FALSE;

    if (args)
        arity = 1;

    for (tp = args; tp && tp->pn_op && (tp->pn_op->op_num == PT_OP_COMMA); tp =
             tp->pn_right)
        arity++;

    for (udf = udfuncs; udf; udf = udf->ud_next)
        if (eq(name, udf->ud_name)) {
            if (arity == udf->ud_arity) {
                break;
            } else {
                found = TRUE;
                rarity = udf->ud_arity;
            }
        }

    if (udf == NULL) {
        if (found)
            fprintf(cp_err,
                    "Warning: the user-defined function %s has %d args\n",
                    name, rarity);
        return (NULL);
    }

    for (s = udf->ud_name; *s; s++)
        ;
    s++;

    /* Now we have to traverse the tree and copy it over,
     * substituting args.
     */
    return (trcopy(udf->ud_text, s, args));
}


/* Copy the tree and replace formal args with the right stuff. The way
 * we know that something might be a formal arg is when it is a dvec
 * with length 0 and a name that isn't "list". I hope nobody calls their
 * formal parameters "list".
 */

static struct pnode *
trcopy(struct pnode *tree, char *args, struct pnode *nn)
{
    struct pnode *pn;
    struct dvec *d;
    char *s;
    int i;

    if (tree->pn_value) {

        d = tree->pn_value;

        if ((d->v_length == 0) && strcmp(d->v_name, "list")) {
            /* Yep, it's a formal parameter. Substitute for it.
             * IMPORTANT: we never free parse trees, so we
             * needn't worry that they aren't trees here.
             */
            s = args;
            i = 1;
            while (*s) {
                if (eq(s, d->v_name))
                    break;
                else
                    i++;
                while (*s++)   /* Get past the last '\0'. */
                    ;
            }

            if (*s)
                return (ntharg(i, nn));
            else
                return (tree);

        } else {

            return (tree);

        }

    } else if (tree->pn_func) {

        pn = alloc(struct pnode);
        pn->pn_use = 0;
        pn->pn_name = NULL;
        pn->pn_value = NULL;
        /* pn_func are pointers to a global constant struct */
        pn->pn_func = tree->pn_func;
        pn->pn_op = NULL;
        pn->pn_left = trcopy(tree->pn_left, args, nn);
        pn->pn_left->pn_use++;
        pn->pn_right = NULL;
        pn->pn_next = NULL;

    } else if (tree->pn_op) {

        pn = alloc(struct pnode);
        pn->pn_use = 0;
        pn->pn_name = NULL;
        pn->pn_value = NULL;
        pn->pn_func = NULL;
        /* pn_op are pointers to a global constant struct */
        pn->pn_op = tree->pn_op;
        pn->pn_left = trcopy(tree->pn_left, args, nn);
        pn->pn_left->pn_use++;
        if (pn->pn_op->op_arity == 2) {
            pn->pn_right = trcopy(tree->pn_right, args, nn);
            pn->pn_right->pn_use++;
        } else {
            pn->pn_right = NULL;
        }
        pn->pn_next = NULL;

    } else {
        fprintf(cp_err, "trcopy: Internal Error: bad parse node\n");
        return (NULL);
    }

    return (pn);
}


/* Find the n'th arg in the arglist, returning NULL if there isn't one.
 * Since comma has such a low priority and associates to the right,
 * we can just follow the right branch of the tree num times.
 * Note that we start at 1 when numbering the args.
 */

static struct pnode *
ntharg(int num, struct pnode *args)
{
    struct pnode *ptry;

    ptry = args;

    if (num > 1)
        while (--num > 0) {
            if (ptry && ptry->pn_op &&
                (ptry->pn_op->op_num != PT_OP_COMMA)) {
                if (num == 1)
                    break;
                else
                    return (NULL);
            }
            ptry = ptry->pn_right;
        }

    if (ptry && ptry->pn_op && (ptry->pn_op->op_num == PT_OP_COMMA))
        ptry = ptry->pn_left;

    return (ptry);
}


void
com_undefine(wordlist *wlist)
{
    struct udfunc *udf, *ludf;

    if (!wlist)
        return;

    if (*wlist->wl_word == '*') {
        for (udf = udfuncs; udf;) {
            struct udfunc *next = udf->ud_next;
            cp_remkword(CT_UDFUNCS, udf->ud_name);
            free_pnode(udf->ud_text);
            free(udf->ud_name);
            free(udf);
            udf = next;
        }
        udfuncs = NULL;
        return;
    }

    for (; wlist; wlist = wlist->wl_next) {
        ludf = NULL;
        for (udf = udfuncs; udf;) {
            struct udfunc *next = udf->ud_next;
            if (eq(wlist->wl_word, udf->ud_name)) {
                if (ludf)
                    ludf->ud_next = udf->ud_next;
                else
                    udfuncs = udf->ud_next;
                cp_remkword(CT_UDFUNCS, wlist->wl_word);
                free_pnode(udf->ud_text);
                free(udf->ud_name);
                free(udf);
            } else {
                ludf = udf;
            }
            udf = next;
        }
    }
}


#ifndef LINT

/* Watch out, this is not at all portable.  It's only here so I can
 * call it from dbx with an int value (all you can give with "call")...
 */

void
ft_pnode(struct pnode *pn)
{
    prtree1(pn, cp_err);
}

#endif
