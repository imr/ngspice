/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Wayne A. Christopher, U. C. Berkeley CAD Group
**********/

/** \file cmath3.c
    \brief functions for the control language parser: divide, comma, power, eq, gt, lt, ge, le, ne

    Routines to do complex mathematical functions. These routines require
    the -lm libraries. We sacrifice a lot of space to be able
    to avoid having to do a seperate call for every vector element,
    but it pays off in time savings.  These routines should never
    allow FPE's to happen.
  
    Complex functions are called as follows:
     cx_something(data, type, length, &newlength, &newtype),
     and return a char * that is cast to complex or double.
*/


#include "ngspice/ngspice.h"
#include "ngspice/cpdefs.h"
#include "ngspice/dvec.h"

#include "cmath.h"
#include "cmath3.h"


static ngcomplex_t *cexp_sp3(ngcomplex_t *c); /* cexp exist's in some newer compiler */
static int cln(ngcomplex_t *c, ngcomplex_t *rv);
static void ctimes(ngcomplex_t *c1, ngcomplex_t *c2, ngcomplex_t *rv);

void *cx_divide(void *data1, void *data2,
        short int datatype1, short int datatype2, int length)
{
    int xrc = 0;
    void *rv;
    double *dd1 = (double *) data1;
    double *dd2 = (double *) data2;
    ngcomplex_t *cc1 = (ngcomplex_t *) data1;
    ngcomplex_t *cc2 = (ngcomplex_t *) data2;
    ngcomplex_t *c, c1, c2;
    int i;

    if ((datatype1 == VF_REAL) && (datatype2 == VF_REAL)) {
        double *d;
        rv = d = alloc_d(length);
        for (i = 0; i < length; i++) {
            rcheck(dd2[i] != 0, "divide");
            d[i] = dd1[i] / dd2[i];
        }
    }
    else {
        rv = c = alloc_c(length);
        for (i = 0; i < length; i++) {
            if (datatype1 == VF_REAL) {
                realpart(c1) = dd1[i];
                imagpart(c1) = 0.0;
            }
            else {
                c1 = cc1[i];
            }
            if (datatype2 == VF_REAL) {
                realpart(c2) = dd2[i];
                imagpart(c2) = 0.0;
            }
            else {
                c2 = cc2[i];
            }
        rcheck((realpart(c2) != 0) || (imagpart(c2) != 0), "divide");
#define xx5 realpart(c1)
#define xx6 imagpart(c1)
        cdiv(xx5, xx6, realpart(c2), imagpart(c2), realpart(c[i]),
                imagpart(c[i]));
        }
    }

EXITPOINT:
    if (xrc != 0) { /* Free resources on error */
        tfree(rv);
        rv = NULL;
    }

    return rv;
} /* end of function cx_divide */



/* Should just use "j( )" */
/* The comma operator. What this does (unless it is part of the argument
 * list of a user-defined function) is arg1 + j(arg2).
 */

void *
cx_comma(void *data1, void *data2, short int datatype1, short int datatype2, int length)
{
    double *dd1 = (double *) data1;
    double *dd2 = (double *) data2;
    ngcomplex_t *cc1 = (ngcomplex_t *) data1;
    ngcomplex_t *cc2 = (ngcomplex_t *) data2;
    ngcomplex_t *c, c1, c2;
    int i;

    c = alloc_c(length);
    for (i = 0; i < length; i++) {
        if (datatype1 == VF_REAL) {
            realpart(c1) = dd1[i];
            imagpart(c1) = 0.0;
        } else {
            c1 = cc1[i];
        }
        if (datatype2 == VF_REAL) {
            realpart(c2) = dd2[i];
            imagpart(c2) = 0.0;
        } else {
            c2 = cc2[i];
        }

        realpart(c[i]) = realpart(c1) + imagpart(c2);
        imagpart(c[i]) = imagpart(c1) + realpart(c2);
    }
    return ((void *) c);
}

void *cx_power(void *data1, void *data2,
        short int datatype1, short int datatype2, int length)
{
    int xrc = 0;
    void *rv;
    double *dd1 = (double *) data1;
    double *dd2 = (double *) data2;

    if ((datatype1 == VF_REAL) && (datatype2 == VF_REAL)) {
        double *d;
        rv = d = alloc_d(length);

        int i;
        for (i = 0; i < length; i++) {
            rcheck((dd1[i] >= 0) || (floor(dd2[i]) == ceil(dd2[i])), "power");
            d[i] = pow(dd1[i], dd2[i]);
        }
    }
    else {
        ngcomplex_t *cc1 = (ngcomplex_t *) data1;
        ngcomplex_t *cc2 = (ngcomplex_t *) data2;
        ngcomplex_t *c, c1, c2, *t;
        rv = c = alloc_c(length);

        int i;
        for (i = 0; i < length; i++) {
            if (datatype1 == VF_REAL) {
                realpart(c1) = dd1[i];
                imagpart(c1) = 0.0;
            }
            else {
                c1 = cc1[i];
            }
            if (datatype2 == VF_REAL) {
                realpart(c2) = dd2[i];
                imagpart(c2) = 0.0;
            }
            else {
                c2 = cc2[i];
            }

            if ((realpart(c1) == 0.0) && (imagpart(c1) == 0.0)) {
                realpart(c[i]) = 0.0;
                imagpart(c[i]) = 0.0;
            }
            else { /* if ((imagpart(c1) != 0.0) && 
                        (imagpart(c2) != 0.0)) */
                ngcomplex_t tmp, tmp2;
                if (cln(&c1, &tmp) != 0) {
                    (void) fprintf(cp_err, "power of 0 + i 0 not allowed.\n");
                    xrc = -1;
                    goto EXITPOINT;
                }
                ctimes(&c2, &tmp, &tmp2);
                t = cexp_sp3(&tmp2);
                c[i] = *t;
                /*
                } else {
                    realpart(c[i]) = pow(realpart(c1), 
                                    realpart(c2)); 
                    imagpart(c[i]) = 0.0;
                */
            }
        }
    }

EXITPOINT:
    if (xrc != 0) { /* Free resources on error */
        txfree(rv);
        rv = NULL;
    }

    return rv;
} /* end of function cx_power */



/* These are unnecessary... Only cx_power uses them... */

static ngcomplex_t *
cexp_sp3(ngcomplex_t *c)
{
    static ngcomplex_t r;
    double d;

    d = exp(realpart(*c));
    realpart(r) = d * cos(imagpart(*c));
    if (imagpart(*c) != 0.0)
        imagpart(r) = d * sin(imagpart(*c));
    else
        imagpart(r) = 0.0;
    return (&r);
}

static int cln(ngcomplex_t *c, ngcomplex_t *rv)
{
    double c_r = c->cx_real;
    double c_i = c->cx_imag;

    if (c_r == 0 && c_i == 0) {
        (void) fprintf(cp_err, "Complex log of 0 + i0 is undefined.\n");
        return -1;
    }

    rv->cx_real = log(cmag(*c));
    if (c_i != 0.0) {
        rv->cx_imag = atan2(c_i, c_r);
    }
    else {
        rv->cx_imag = 0.0;
    }
    return 0;
} /* end of functon cln */



static void ctimes(ngcomplex_t *c1, ngcomplex_t *c2, ngcomplex_t *rv)
{
    rv->cx_real = realpart(*c1) * realpart(*c2) -
               imagpart(*c1) * imagpart(*c2);
    rv->cx_imag = imagpart(*c1) * realpart(*c2) +
               realpart(*c1) * imagpart(*c2);
    return;
}



/* Now come all the relational and logical functions. It's overkill to put
 * them here, but... Note that they always return a real value, with the
 * result the same length as the arguments.
 */

void *
cx_eq(void *data1, void *data2, short int datatype1, short int datatype2, int length)
{
    double *dd1 = (double *) data1;
    double *dd2 = (double *) data2;
    double *d;
    ngcomplex_t *cc1 = (ngcomplex_t *) data1;
    ngcomplex_t *cc2 = (ngcomplex_t *) data2;
    ngcomplex_t c1, c2;
    int i;

    d = alloc_d(length);
    if ((datatype1 == VF_REAL) && (datatype2 == VF_REAL)) {
        for (i = 0; i < length; i++)
            if (dd1[i] == dd2[i])
                d[i] = 1.0;
            else
                d[i] = 0.0;
    } else {
        for (i = 0; i < length; i++) {
            if (datatype1 == VF_REAL) {
                realpart(c1) = dd1[i];
                imagpart(c1) = 0.0;
            } else {
                c1 = cc1[i];
            }
            if (datatype2 == VF_REAL) {
                realpart(c2) = dd2[i];
                imagpart(c2) = 0.0;
            } else {
                c2 = cc2[i];
            }
            d[i] = ((realpart(c1) == realpart(c2)) &&
                (imagpart(c1) == imagpart(c2)));
        }
    }
    return ((void *) d);
}

void *
cx_gt(void *data1, void *data2, short int datatype1, short int datatype2, int length)
{
    double *dd1 = (double *) data1;
    double *dd2 = (double *) data2;
    double *d;
    ngcomplex_t *cc1 = (ngcomplex_t *) data1;
    ngcomplex_t *cc2 = (ngcomplex_t *) data2;
    ngcomplex_t c1, c2;
    int i;

    d = alloc_d(length);
    if ((datatype1 == VF_REAL) && (datatype2 == VF_REAL)) {
        for (i = 0; i < length; i++)
            if (dd1[i] > dd2[i])
                d[i] = 1.0;
            else
                d[i] = 0.0;
    } else {
        for (i = 0; i < length; i++) {
            if (datatype1 == VF_REAL) {
                realpart(c1) = dd1[i];
                imagpart(c1) = 0.0;
            } else {
                c1 = cc1[i];
            }
            if (datatype2 == VF_REAL) {
                realpart(c2) = dd2[i];
                imagpart(c2) = 0.0;
            } else {
                c2 = cc2[i];
            }
            d[i] = ((realpart(c1) > realpart(c2)) &&
                (imagpart(c1) > imagpart(c2)));
        }
    }
    return ((void *) d);
}

void *
cx_lt(void *data1, void *data2, short int datatype1, short int datatype2, int length)
{
    double *dd1 = (double *) data1;
    double *dd2 = (double *) data2;
    double *d;
    ngcomplex_t *cc1 = (ngcomplex_t *) data1;
    ngcomplex_t *cc2 = (ngcomplex_t *) data2;
    ngcomplex_t c1, c2;
    int i;

    d = alloc_d(length);
    if ((datatype1 == VF_REAL) && (datatype2 == VF_REAL)) {
        for (i = 0; i < length; i++)
            if (dd1[i] < dd2[i])
                d[i] = 1.0;
            else
                d[i] = 0.0;
    } else {
        for (i = 0; i < length; i++) {
            if (datatype1 == VF_REAL) {
                realpart(c1) = dd1[i];
                imagpart(c1) = 0.0;
            } else {
                c1 = cc1[i];
            }
            if (datatype2 == VF_REAL) {
                realpart(c2) = dd2[i];
                imagpart(c2) = 0.0;
            } else {
                c2 = cc2[i];
            }
            d[i] = ((realpart(c1) < realpart(c2)) &&
                (imagpart(c1) < imagpart(c2)));
        }
    }
    return ((void *) d);
}

void *
cx_ge(void *data1, void *data2, short int datatype1, short int datatype2, int length)
{
    double *dd1 = (double *) data1;
    double *dd2 = (double *) data2;
    double *d;
    ngcomplex_t *cc1 = (ngcomplex_t *) data1;
    ngcomplex_t *cc2 = (ngcomplex_t *) data2;
    ngcomplex_t c1, c2;
    int i;

    d = alloc_d(length);
    if ((datatype1 == VF_REAL) && (datatype2 == VF_REAL)) {
        for (i = 0; i < length; i++)
            if (dd1[i] >= dd2[i])
                d[i] = 1.0;
            else
                d[i] = 0.0;
    } else {
        for (i = 0; i < length; i++) {
            if (datatype1 == VF_REAL) {
                realpart(c1) = dd1[i];
                imagpart(c1) = 0.0;
            } else {
                c1 = cc1[i];
            }
            if (datatype2 == VF_REAL) {
                realpart(c2) = dd2[i];
                imagpart(c2) = 0.0;
            } else {
                c2 = cc2[i];
            }
            d[i] = ((realpart(c1) >= realpart(c2)) &&
                (imagpart(c1) >= imagpart(c2)));
        }
    }
    return ((void *) d);
}

void *
cx_le(void *data1, void *data2, short int datatype1, short int datatype2, int length)
{
    double *dd1 = (double *) data1;
    double *dd2 = (double *) data2;
    double *d;
    ngcomplex_t *cc1 = (ngcomplex_t *) data1;
    ngcomplex_t *cc2 = (ngcomplex_t *) data2;
    ngcomplex_t c1, c2;
    int i;

    d = alloc_d(length);
    if ((datatype1 == VF_REAL) && (datatype2 == VF_REAL)) {
        for (i = 0; i < length; i++)
            if (dd1[i] <= dd2[i])
                d[i] = 1.0;
            else
                d[i] = 0.0;
    } else {
        for (i = 0; i < length; i++) {
            if (datatype1 == VF_REAL) {
                realpart(c1) = dd1[i];
                imagpart(c1) = 0.0;
            } else {
                c1 = cc1[i];
            }
            if (datatype2 == VF_REAL) {
                realpart(c2) = dd2[i];
                imagpart(c2) = 0.0;
            } else {
                c2 = cc2[i];
            }
            d[i] = ((realpart(c1) <= realpart(c2)) &&
                (imagpart(c1) <= imagpart(c2)));
        }
    }
    return ((void *) d);
}

void *
cx_ne(void *data1, void *data2, short int datatype1, short int datatype2, int length)
{
    double *dd1 = (double *) data1;
    double *dd2 = (double *) data2;
    double *d;
    ngcomplex_t *cc1 = (ngcomplex_t *) data1;
    ngcomplex_t *cc2 = (ngcomplex_t *) data2;
    ngcomplex_t c1, c2;
    int i;

    d = alloc_d(length);
    if ((datatype1 == VF_REAL) && (datatype2 == VF_REAL)) {
        for (i = 0; i < length; i++)
            if (dd1[i] != dd2[i])
                d[i] = 1.0;
            else
                d[i] = 0.0;
    } else {
        for (i = 0; i < length; i++) {
            if (datatype1 == VF_REAL) {
                realpart(c1) = dd1[i];
                imagpart(c1) = 0.0;
            } else {
                c1 = cc1[i];
            }
            if (datatype2 == VF_REAL) {
                realpart(c2) = dd2[i];
                imagpart(c2) = 0.0;
            } else {
                c2 = cc2[i];
            }
            d[i] = ((realpart(c1) != realpart(c2)) &&
                (imagpart(c1) != imagpart(c2)));
        }
    }
    return ((void *) d);
}

