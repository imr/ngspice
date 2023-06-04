/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Wayne A. Christopher, U. C. Berkeley CAD Group
**********/

/*
 * Convert a parse tree to a list of data vectors.
 */

#include "ngspice/ngspice.h"

#include <setjmp.h>
#include <signal.h>

#include "ngspice/ftedefs.h"
#include "ngspice/dvec.h"

#include "evaluate.h"

#include "ngspice/sim.h"  /* To get SV_VOLTAGE definition */


static void sig_matherr(void);
static struct dvec *apply_func(struct func *func, struct pnode *arg);
static struct dvec *ft_ternary(struct pnode *node);
static char *mkcname(char what, char *v1, char *v2);


/* We are careful here to catch SIGILL and recognise them as math errors.
 * The only trouble is that the (void) signal handler we installed before will
 * be lost, but that's no great loss.
 */

static JMP_BUF matherrbuf;

static void
sig_matherr(void)
{
    fprintf(cp_err, "Error: argument out of range for math function\n");
    LONGJMP(matherrbuf, 1);
}


/* Note that ft_evaluate will return NULL on invalid expressions. */
/* va: NOTE: ft_evaluate returns a new vector for expressions (func, op, ...)
   and an existing vector (node->pn_value) when node->pn_value != NULL.
   For garbage collection caller must vec_free() expression-vector. */
struct dvec *
ft_evaluate(struct pnode *node)
{
    struct dvec *d = NULL;

    if (!node)
        d = NULL;
    else if (node->pn_value)
        d = node->pn_value;
    else if (node->pn_func)
        d = apply_func(node->pn_func, node->pn_left);
    else if (node->pn_op) {
        if (node->pn_op->op_arity == 1)
            d = node->pn_op->op_func.unary (node->pn_left);
        else if (node->pn_op->op_arity == 2) {
            if (node->pn_op->op_num == PT_OP_TERNARY)
                d = ft_ternary(node);
            else
                d = node->pn_op->op_func.binary (node->pn_left, node->pn_right);
        }
    } else {
        fprintf(cp_err, "ft_evaluate: Internal Error: bad node\n");
        d = NULL;
    }

    if (d == NULL) {
        if (node && node->pn_name) {
            fprintf(stderr, "  in term: %s\n\n", node->pn_name);
        }
        return NULL;
    }

    if (node->pn_name && !ft_evdb && d && !d->v_link2) {
        if (d->v_name)
            tfree(d->v_name); /* patch by Stefan Jones */
        d->v_name = copy(node->pn_name);
    }

    if (!d->v_length) {
        fprintf(cp_err, "Error: no such vector %s\n", d->v_name);
        return (NULL);
    } else {
        return (d);
    }
}


static struct dvec *
ft_ternary(struct pnode *node)
{
    struct dvec *v, *d, *cond;
    struct pnode *arg;
    int c;

    if (!node->pn_right->pn_op || node->pn_right->pn_op->op_func.binary != op_comma)
    {
        fprintf(cp_err, "Error: ft_ternary(), daemons ...\n");
        return NULL;
    }

    cond = ft_evaluate(node->pn_left);

    if (cond->v_link2) {
        fprintf(cp_err, "Error: ft_ternary(), whats that ?\n");
        return NULL;
    }

    if (cond->v_numdims != 1) {
        fprintf(cp_err, "Error: ft_ternary(), condition must be scalar, but numdims=%d\n",
                cond->v_numdims);
        return NULL;
    }

    if (cond->v_length != 1) {
        fprintf(cp_err, "Error: ft_ternary(), condition must be scalar, but length=%d\n",
                cond->v_length);
        return NULL;
    }

    c = isreal(cond)
        ? (cond->v_realdata[0] != 0.0)
        : ((realpart(cond->v_compdata[0]) != 0.0) ||
           (imagpart(cond->v_compdata[0]) != 0.0));

    arg = c
        ? node->pn_right->pn_left
        : node->pn_right->pn_right;

    v = ft_evaluate(arg);
    d = vec_copy(v);
    vec_new(d);

    if (!arg->pn_value && v)
        vec_free(v);
    if (!node->pn_left->pn_value && cond)
        vec_free(cond);

    return d;
}


/* Operate on two vectors, and return a third with the data, length, and flags
 * fields filled in. Add it to the current plot and get rid of the two args.
 */

static void *
doop_funcall(
    void * (*func) (void *data1, void *data2,
                    short int datatype1, short int datatype2,
                    int length),
    void *data1, void *data2,
    short int datatype1, short int datatype2,
    int length)
{
    void *data;

    /* Some of the math routines generate SIGILL if the argument is
     * out of range.  Catch this here.
     */

    if (SETJMP(matherrbuf, 1)) {
        return (NULL);
    }

    (void) signal(SIGILL, (SIGNAL_FUNCTION) sig_matherr);

    data = func(data1, data2, datatype1, datatype2, length);

    /* Back to normal */
    (void) signal(SIGILL, SIG_DFL);

    return data;
}


static struct dvec *
doop(char what,
     void * (*func) (void *data1, void *data2,
                     short int datatype1, short int datatype2,
                     int length),
     struct pnode *arg1,
     struct pnode *arg2)
{
    struct dvec *v1, *v2, *res;
    ngcomplex_t *c1 = NULL, *c2 = NULL, lc;
    double *d1 = NULL, *d2 = NULL, ld;
    int length = 0, i, longer;
    void *data;
    bool free1 = FALSE, free2 = FALSE, relflag = FALSE;

    v1 = ft_evaluate(arg1);
    v2 = ft_evaluate(arg2);
    if (!v1 || !v2)
        return (NULL);

    /* Now the question is, what do we do when one or both of these
     * has more than one vector?  This is definitely not a good
     * thing.  For the time being don't do anything.
     */
    if (v1->v_link2 || v2->v_link2) {
        fprintf(cp_err, "Warning: no operations on wildcards yet.\n");
        if (v1->v_link2 && v2->v_link2)
            fprintf(cp_err, "\t(You couldn't do that one anyway)\n");
        return (NULL);
    }

    /* How do we handle operations on multi-dimensional vectors?
     * For now, we only allow operations between one-D vectors,
     * equivalently shaped multi-D vectors, or a multi-D vector and
     * a one-D vector.  It's not at all clear what to do in the other cases.
     * So only check shape requirement if it is an operation between two multi-D
     * arrays.
     */
    if ((v1->v_numdims > 1) && (v2->v_numdims > 1)) {
        if (v1->v_numdims != v2->v_numdims) {
            fprintf(cp_err,
                    "Warning: operands %s and %s have incompatible shapes.\n",
                    v1->v_name, v2->v_name);
            return (NULL);
        }
        for (i = 1; i < v1->v_numdims; i++)
            if ((v1->v_dims[i] != v2->v_dims[i])) {
                fprintf(cp_err,
                        "Warning: operands %s and %s have incompatible shapes.\n",
                        v1->v_name, v2->v_name);
                return (NULL);
            }
    }

    /* This is a bad way to do this. */
    switch (what) {
    case '=':
    case '>':
    case '<':
    case 'G':
    case 'L':
    case 'N':
    case '&':
    case '|':
    case '~':
        relflag = TRUE;
    }

    /* Type checking is done later */

    /* Make sure we have data of the same length. */
    length = ((v1->v_length > v2->v_length) ? v1->v_length : v2->v_length);
    if (v1->v_length < length) {
        longer = 2;
        free1 = TRUE;
        if (isreal(v1)) {
            ld = 0.0;
            d1 = TMALLOC(double, length);
            for (i = 0; i < v1->v_length; i++)
                d1[i] = v1->v_realdata[i];
            if (i > 0)
                ld = v1->v_realdata[i - 1];
            for (; i < length; i++)
                d1[i] = ld;
        } else {
            realpart(lc) = 0.0;
            imagpart(lc) = 0.0;
            c1 = TMALLOC(ngcomplex_t, length);
            for (i = 0; i < v1->v_length; i++)
                c1[i] = v1->v_compdata[i];
            if (i > 0)
                lc = v1->v_compdata[i - 1];
            for (; i < length; i++)
                c1[i] = lc;
        }
    } else {
        longer = 0;
        if (isreal(v1))
            d1 = v1->v_realdata;
        else
            c1 = v1->v_compdata;
    }

    if (v2->v_length < length) {
        longer = 1;
        free2 = TRUE;
        if (isreal(v2)) {
            ld = 0.0;
            d2 = TMALLOC(double, length);
            for (i = 0; i < v2->v_length; i++)
                d2[i] = v2->v_realdata[i];
            if (i > 0)
                ld = v2->v_realdata[i - 1];
            for (; i < length; i++)
                d2[i] = ld;
        } else {
            realpart(lc) = 0.0;
            imagpart(lc) = 0.0;
            c2 = TMALLOC(ngcomplex_t, length);
            for (i = 0; i < v2->v_length; i++)
                c2[i] = v2->v_compdata[i];
            if (i > 0)
                lc = v2->v_compdata[i - 1];
            for (; i < length; i++)
                c2[i] = lc;
        }
    } else {
        if (isreal(v2))
            d2 = v2->v_realdata;
        else
            c2 = v2->v_compdata;
    }

    /* Now pass the vectors to the appropriate function. */
    data = doop_funcall
        (func,
         isreal(v1) ? (void *) d1 : (void *) c1,
         isreal(v2) ? (void *) d2 : (void *) c2,
         isreal(v1) ? VF_REAL : VF_COMPLEX,
         isreal(v2) ? VF_REAL : VF_COMPLEX,
         length);

    if (!data)
        return (NULL);
    /* Make up the new vector. */
    if (relflag || (isreal(v1) && isreal(v2) && (func != cx_comma))) {
        res = dvec_alloc(mkcname(what, v1->v_name, v2->v_name),
                         SV_NOTYPE,
                         (v1->v_flags | v2->v_flags | VF_REAL) & ~VF_COMPLEX,
                         length, data);
    } else {
        res = dvec_alloc(mkcname(what, v1->v_name, v2->v_name),
                         SV_NOTYPE,
                         (v1->v_flags | v2->v_flags | VF_COMPLEX) & ~VF_REAL,
                         length, data);
    }

    /* This is a non-obvious thing */

    if (v1->v_scale != v2->v_scale) {
        switch (longer) {
        case 0:
            if (!v1->v_scale)
                res->v_scale = v2->v_scale;
            else if (!v2->v_scale)
                res->v_scale = v1->v_scale;
            else
                fprintf(cp_err,
                         "Warning: scales of %s and %s are different.\n",
                        v1->v_name, v2->v_name);
            res->v_scale = v1->v_scale; // Do something!
            break;
        case 1:
            res->v_scale = v1->v_scale;
            break;
        case 2:
            res->v_scale = v2->v_scale;
            break;
        }
    } else {
        res->v_scale = v1->v_scale;
    }

    /* Copy a few useful things */
    res->v_defcolor = v1->v_defcolor;
    res->v_gridtype = v1->v_gridtype;
    res->v_plottype = v1->v_plottype;

    /* Copy dimensions. */
    if (v1->v_numdims > v2->v_numdims) {
        res->v_numdims = v1->v_numdims;
        for (i = 0; i < v1->v_numdims; i++)
            res->v_dims[i] = v1->v_dims[i];
    } else {
        res->v_numdims = v2->v_numdims;
        for (i = 0; i < v2->v_numdims; i++)
            res->v_dims[i] = v2->v_dims[i];
    }

    /* ** Type checking for multiplication and division of vectors **
     * Determines the units resulting from the operation.
     * A.Roldán
     */
    switch (what)
    {
    case '*':  /* Multiplication of two vectors */
        switch (v1->v_type)
        {
        case SV_VOLTAGE:
            switch (v2->v_type)
            {
            case SV_VOLTAGE:
                res->v_type = SV_VOLTAGE;
                break;
            case SV_CURRENT:
                res->v_type = SV_POWER;
                break;
            default:
                break;
            }
            break;

        case SV_CURRENT:
            switch (v2->v_type)
            {
            case SV_VOLTAGE:
                res->v_type = SV_POWER;
                break;
            case SV_CURRENT:
                res->v_type = SV_CURRENT;
                break;
            default:
                break;
            }
            break;

        default:
            break;
        }
        break;
    case '/':   /* division of two vectors */
        switch (v1->v_type)
        {
        case SV_VOLTAGE:
            switch (v2->v_type)
            {
            case SV_VOLTAGE:
                res->v_type = SV_NOTYPE;
                break;
            case SV_CURRENT:
                res->v_type = SV_IMPEDANCE;
                break;
            default:
                break;
            }
            break;

        case SV_CURRENT:
            switch (v2->v_type)
            {
            case SV_VOLTAGE:
                res->v_type = SV_ADMITTANCE;
                break;
            case SV_CURRENT:
                res->v_type = SV_NOTYPE;
                break;
            default:
                break;
            }
            break;

        default:
            break;
        }

    default:
        break;
    }

    vec_new(res);

    /* Free the temporary data areas we used, if we allocated any. */
    if (free1) {
        if (isreal(v1))
            tfree(d1);
        else
            tfree(c1);
    }

    if (free2) {
        if (isreal(v2))
            tfree(d2);
        else
            tfree(c2);
    }

    /* va: garbage collection */
    if (arg1->pn_value == NULL && v1 != NULL)
        vec_free(v1);
    if (arg2->pn_value == NULL && v2 != NULL)
        vec_free(v2);

    return (res);
}


/* The binary operations. */
struct dvec *
op_plus(struct pnode *arg1, struct pnode *arg2)
{
    return (doop('+', cx_plus, arg1, arg2));
}

struct dvec *
op_minus(struct pnode *arg1, struct pnode *arg2)
{
    return (doop('-', cx_minus, arg1, arg2));
}

struct dvec *
op_comma(struct pnode *arg1, struct pnode *arg2)
{
    return (doop(',', cx_comma, arg1, arg2));
}

struct dvec *
op_times(struct pnode *arg1, struct pnode *arg2)
{
    return (doop('*', cx_times, arg1, arg2));
}

struct dvec *
op_mod(struct pnode *arg1, struct pnode *arg2)
{
    return (doop('%', cx_mod, arg1, arg2));
}

struct dvec *
op_divide(struct pnode *arg1, struct pnode *arg2)
{
    return (doop('/', cx_divide, arg1, arg2));
}

struct dvec *
op_power(struct pnode *arg1, struct pnode *arg2)
{
    return (doop('^', cx_power, arg1, arg2));
}

struct dvec *
op_eq(struct pnode *arg1, struct pnode *arg2)
{
    return (doop('=', cx_eq, arg1, arg2));
}

struct dvec *
op_gt(struct pnode *arg1, struct pnode *arg2)
{
    return (doop('>', cx_gt, arg1, arg2));
}

struct dvec *
op_lt(struct pnode *arg1, struct pnode *arg2)
{
    return (doop('<', cx_lt, arg1, arg2));
}

struct dvec *
op_ge(struct pnode *arg1, struct pnode *arg2)
{
    return (doop('G', cx_ge, arg1, arg2));
}

struct dvec *
op_le(struct pnode *arg1, struct pnode *arg2)
{
    return (doop('L', cx_le, arg1, arg2));
}

struct dvec *
op_ne(struct pnode *arg1, struct pnode *arg2)
{
    return (doop('N', cx_ne, arg1, arg2));
}

struct dvec *
op_and(struct pnode *arg1, struct pnode *arg2)
{
    return (doop('&', cx_and, arg1, arg2));
}

struct dvec *
op_or(struct pnode *arg1, struct pnode *arg2)
{
    return (doop('|', cx_or, arg1, arg2));
}


/* This is an odd operation.  The first argument is the name of a vector, and
 * the second is a range in the scale, so that v(1)[[10, 20]] gives all the
 * values of v(1) for which the TIME value is between 10 and 20.  If there is
 * one argument it picks out the values which have that scale value.
 * NOTE that we totally ignore multi-dimensionality here -- the result is
 * a 1-dim vector.
 */

struct dvec *
op_range(struct pnode *arg1, struct pnode *arg2)
{
    struct dvec *v, *ind, *res, *scale;
    double up, low, td;
    int len, i, j;
    bool rev = FALSE;

    v = ft_evaluate(arg1);
    ind = ft_evaluate(arg2);
    if (!v || !ind)
        return (NULL);

    scale = v->v_scale;
    if (!scale)
        scale = v->v_plot->pl_scale;

    if (!scale) {
        fprintf(cp_err, "Error: no scale for vector %s\n", v->v_name);
        return (NULL);
    }

    if (ind->v_length != 1) {
        fprintf(cp_err, "Error: strange range specification\n");
        return (NULL);
    }

    if (isreal(ind)) {
        up = low = *ind->v_realdata;
    } else {
        up = imagpart(ind->v_compdata[0]);
        low = realpart(ind->v_compdata[0]);
    }

    if (up < low) {
        SWAP(double, up, low);
        rev = TRUE;
    }

    for (i = len = 0; i < scale->v_length; i++) {
        td = isreal(scale) ? scale->v_realdata[i] :
            realpart(scale->v_compdata[i]);
        if ((td <= up) && (td >= low))
            len++;
    }

    res = dvec_alloc(mkcname('R', v->v_name, ind->v_name),
                     v->v_type,
                     v->v_flags,
                     len, NULL);

    res->v_gridtype = v->v_gridtype;
    res->v_plottype = v->v_plottype;
    res->v_defcolor = v->v_defcolor;
    res->v_scale = /* nscale; */ scale;
    /* Dave says get rid of this
       res->v_numdims = v->v_numdims;
       for (i = 0; i < v->v_numdims; i++)
       res->v_dims[i] = v->v_dims[i];
    */
    res->v_numdims = 1;
    res->v_dims[0] = len;

    /* Toss in the data */

    j = 0;
    for (i = (rev ? v->v_length - 1 : 0);
         i != (rev ? -1 : v->v_length);
         rev ? i-- : i++)
    {
        td = isreal(scale) ? scale->v_realdata[i] :
            realpart(scale->v_compdata[i]);
        if ((td <= up) && (td >= low)) {
            if (isreal(res)) {
                res->v_realdata[j] = v->v_realdata[i];
            } else {
                res->v_compdata[j] = v->v_compdata[i];
            }
            j++;
        }
    }

    if (j != len)
        fprintf(cp_err, "Error: something funny..\n");

    /* Note that we DON'T do a vec_new, since we want this vector to be
     * invisible to everybody except the result of this operation.
     * Doing this will cause a lot of core leaks, though. XXX
     */

    vec_new(res);

    /* va: garbage collection */
    if (arg1->pn_value == NULL && v != NULL)
        vec_free(v);
    if (arg2->pn_value == NULL && ind != NULL)
        vec_free(ind);

    return (res);
}


/* This is another operation we do specially -- if the argument is a vector of
 * dimension n, n > 0, the result will be either a vector of dimension n - 1,
 * or a vector of dimension n with only a certain range of vectors present.
 */

struct dvec *
op_ind(struct pnode *arg1, struct pnode *arg2)
{
    struct dvec *v, *ind, *res;
    int length, newdim, i, j, k, up, down;
    int majsize, blocksize;
    bool rev = FALSE;

    v = ft_evaluate(arg1);
    ind = ft_evaluate(arg2);
    if (!v || !ind)
        return (NULL);

    /* First let's check to make sure that the vector is consistent */
    if (v->v_numdims > 1) {
        for (i = 0, j = 1; i < v->v_numdims; i++)
            j *= v->v_dims[i];
        if (v->v_length != j) {
            fprintf(cp_err,
                    "op_ind: Internal Error: len %d should be %d\n",
                    v->v_length, j);
            return (NULL);
        }
    } else {
        /* Just in case we were sloppy */
        v->v_numdims = 1;
        v->v_dims[0] = v->v_length;
        if (v->v_length <= 1) {
            fprintf(cp_err, "Error: nostrchring on a scalar (%s)\n",
                    v->v_name);
            return (NULL);
        }
    }

    if (ind->v_length != 1) {
        fprintf(cp_err, "Error:strchr %s is not of length 1\n",
                ind->v_name);
        return (NULL);
    }

    majsize = v->v_dims[0];
    blocksize = v->v_length / majsize;

    /* Now figure out if we should put the dim down by one.  Because of the
     * way we parse the strchr, we figure that if the value is complex
     * (e.g, "[1,2]"), the guy meant a range.  This is sort of bad though.
     */
    if (isreal(ind)) {
        newdim = v->v_numdims - 1;
        down = up = (int)floor(ind->v_realdata[0] + 0.5);
    } else {
        newdim = v->v_numdims;
        down = (int)floor(realpart(ind->v_compdata[0]) + 0.5);
        up = (int)floor(imagpart(ind->v_compdata[0]) + 0.5);
    }
    if (up < down) {
        SWAP(int, up, down);
        rev = TRUE;
    }
    if (up < 0) {
        fprintf(cp_err, "Warning: upper limit %d should be 0\n", up);
        up = 0;
    }
    if (up >= majsize) {
        fprintf(cp_err, "Warning: upper limit %d should be %d\n", up,
                majsize - 1);
        up = majsize - 1;
    }
    if (down < 0) {
        fprintf(cp_err, "Warning: lower limit %d should be 0\n", down);
        down = 0;
    }
    if (down >= majsize) {
        fprintf(cp_err, "Warning: lower limit %d should be %d\n", down,
                majsize - 1);
        down = majsize - 1;
    }

    if (up == down)
        length = blocksize;
    else
        length = blocksize * (up - down + 1);

    /* Make up the new vector. */
    res = dvec_alloc(mkcname('[', v->v_name, ind->v_name),
                     v->v_type,
                     v->v_flags,
                     length, NULL);

    res->v_defcolor = v->v_defcolor;
    res->v_gridtype = v->v_gridtype;
    res->v_plottype = v->v_plottype;
    res->v_numdims = newdim;
    if (up != down) {
        for (i = 0; i < newdim; i++)
            res->v_dims[i] = v->v_dims[i];
        res->v_dims[0] = up - down + 1;
    } else {
        for (i = 0; i < newdim; i++)
            res->v_dims[i] = v->v_dims[i + 1];
    }

    /* And toss in the new data */
    for (j = 0; j < up - down + 1; j++) {
        if (rev)
            k = (up - down) - j;
        else
            k = j;
        for (i = 0; i < blocksize; i++)
            if (isreal(res)) {
                res->v_realdata[k * blocksize + i] =
                    v->v_realdata[(down + j) * blocksize + i];
            } else {
                res->v_compdata[k * blocksize + i] =
                    v->v_compdata[(down + j) * blocksize + i];
            }
    }

    /* This is a problem -- the old scale will be no good.  I guess we
     * should make an altered copy of the old scale also.
     */
    /* Even though the old scale is no good and we should somehow decide
     * on a new scale, using the vector as its own scale is not the
     * solution.
     */
    /*
     * res->v_scale = res;
     */

    vec_new(res);

    /* va: garbage collection */
    if (arg1->pn_value == NULL && v != NULL)
        vec_free(v);
    if (arg2->pn_value == NULL && ind != NULL)
        vec_free(ind);

    return (res);
}


/* Apply a function to an argument. Complex functions are called as follows:
 *  cx_something(data, type, length, &newlength, &newtype),
 *  and returns a void * that is cast to complex or double.
 */

static void *
apply_func_funcall(struct func *func, struct dvec *v, int *newlength, short int *newtype)
{
    void *data;

    /* Some of the math routines generate SIGILL if the argument is
     * out of range.  Catch this here.
     */

    if (SETJMP(matherrbuf, 1)) {
        (void) signal(SIGILL, SIG_DFL);
        return (NULL);
    }

    (void) signal(SIGILL, (SIGNAL_FUNCTION) sig_matherr);

    /* Modified for passing necessary parameters to the derive function - A.Roldan */

    if (eq(func->fu_name, "interpolate") || eq(func->fu_name, "deriv") || eq(func->fu_name, "group_delay")
        || eq(func->fu_name, "fft") || eq(func->fu_name, "ifft") || eq(func->fu_name, "integ"))
    {
        void * (*f) (void *data, short int type, int length,
                     int *newlength, short int *newtype,
                     struct plot *, struct plot *, int) =
            (void * (*) (void *, short int, int, int *, short int *, struct plot *, struct plot *, int)) (void *)func->fu_func;
        data = f
            (isreal(v) ? (void *) v->v_realdata : (void *) v->v_compdata,
             (short) (isreal(v) ? VF_REAL : VF_COMPLEX),
             v->v_length,
             newlength, newtype,
             v->v_plot, plot_cur, v->v_dims[0]);
    } else {
        data = func->fu_func
            (isreal(v) ? (void *) v->v_realdata : (void *) v->v_compdata,
             (short) (isreal(v) ? VF_REAL : VF_COMPLEX),
             v->v_length,
             newlength, newtype);
    }

    /* Back to normal */
    (void) signal(SIGILL, SIG_DFL);

    return data;
}


static struct dvec *
apply_func(struct func *func, struct pnode *arg)
{
    struct dvec *v, *t, *newv = NULL, *end = NULL;
    int len, i;
    short type;
    void *data;

    /* Special case. This is not good -- happens when vm(), etc are used
     * and it gets caught as a user-definable function.  Usually v()
     * is caught in the parser.
     */
    if (!func->fu_func) {
        if (!arg->pn_value /* || (arg->pn_value->v_length != 1) XXX */) {
            fprintf(cp_err, "Error: bad v() syntax\n");
            return (NULL);
        }
        /* try not using the current plot, but the plot set in the arg... vector */
        if(arg->pn_value->v_plot && arg->pn_value->v_plot->pl_typename)
            t = vec_fromplot(arg->pn_value->v_name, get_plot(arg->pn_value->v_plot->pl_typename));
        else
            t = vec_fromplot(arg->pn_value->v_name, plot_cur);
        if (!t) {
            fprintf(cp_err, "Error: no such vector %s\n", arg->pn_value->v_name);
            return (NULL);
        }
        t = vec_copy(t);
        vec_new(t);
        return (t);
    }
    v = ft_evaluate(arg);
    if (v == NULL)
        return (NULL);


    for (; v; v = v->v_link2) {

        char *name;

        data = apply_func_funcall(func, v, &len, &type);

        if (!data)
            return (NULL);

#ifdef FTEDEBUG
        if (ft_evdb)
            fprintf(cp_err,
                    "apply_func: func %s to %s len %d, type %d\n",
                    func->fu_name, v->v_name, len, type);
#endif

        if (eq(func->fu_name, "minus"))
            name = mkcname('a', func->fu_name, v->v_name);
        else if (eq(func->fu_name, "not"))
            name = mkcname('c', func->fu_name, v->v_name);
        else
            name = mkcname('b', v->v_name, NULL);

        t = dvec_alloc(name,
                       v->v_type, /* This is strange too. */
                       (v->v_flags & ~VF_COMPLEX & ~VF_REAL &
                        ~VF_PERMANENT & ~VF_MINGIVEN & ~VF_MAXGIVEN) | type,
                       len, data);

        t->v_scale = v->v_scale;

        /* Copy a few useful things */
        t->v_defcolor = v->v_defcolor;
        t->v_gridtype = v->v_gridtype;
        t->v_plottype = v->v_plottype;
        t->v_numdims = v->v_numdims;
        for (i = 0; i < t->v_numdims; i++)
            t->v_dims[i] = v->v_dims[i];

        vec_new(t);

        /* try to figure out the v_type, depending on the function */
        if (eq(func->fu_name, "cph") || eq(func->fu_name, "ph"))
            t->v_type = SV_PHASE;
        else if (eq(func->fu_name, "db"))
            t->v_type = SV_DB;

        if (end)
            end->v_link2 = t;
        else
            newv = t;
        end = t;
    }

    /* va: garbage collection */
    if (arg->pn_value == NULL && v != NULL)
        vec_free(v);

    return (newv);
}


/* The unary minus operation. */

struct dvec *
op_uminus(struct pnode *arg)
{
    return (apply_func(&func_uminus, arg));
}

struct dvec *
op_not(struct pnode *arg)
{
    return (apply_func(&func_not, arg));
}


/* Create a reasonable name for the result of a function application, etc.
 * The what values 'a' and 'b' mean "make a function name" and "make a
 * unary minus", respectively.
 */

static char *
mkcname(char what, char *v1, char *v2)
{
    switch (what) {
    case 'a':
        return tprintf("%s(%s)", v1, v2);
    case 'b':
        return tprintf("-(%s)", v1);
    case 'c':
        return tprintf("~(%s)", v1);
    case '[':
        return tprintf("%s[%s]", v1, v2);
    case 'R':
        return tprintf("%s[[%s]]", v1, v2);
    default:
        return tprintf("(%s)%c(%s)", v1, what, v2);
    }
}
