/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Wayne A. Christopher, U. C. Berkeley CAD Group
**********/

/*
 * Routines to do complex mathematical functions. These routines require
 * the -lm libraries. We sacrifice a lot of space to be able
 * to avoid having to do a seperate call for every vector element,
 * but it pays off in time savings.  These routines should never
 * allow FPE's to happen.
 *
 * Complex functions are called as follows:
 *  cx_something(data, type, length, &newlength, &newtype),
 *  and return a char * that is cast to complex or double.
 */

#include <ngspice.h>
#include <cpdefs.h>
#include <dvec.h>

#include "cmath.h"
#include "cmath3.h"


static complex *cexp(complex *c);
static complex *cln(complex *c);
static complex *ctimes(complex *c1, complex *c2);

void *
cx_divide(void *data1, void *data2, short int datatype1, short int datatype2, int length)
{
    double *dd1 = (double *) data1;
    double *dd2 = (double *) data2;
    double *d;
    complex *cc1 = (complex *) data1;
    complex *cc2 = (complex *) data2;
    complex *c, c1, c2;
    int i;

    if ((datatype1 == VF_REAL) && (datatype2 == VF_REAL)) {
        d = alloc_d(length);
        for (i = 0; i < length; i++) {
            rcheck(dd2[i] != 0, "divide");
            d[i] = dd1[i] / dd2[i];
        }
        return ((void *) d);
    } else {
        c = alloc_c(length);
        for (i = 0; i < length; i++) {
            if (datatype1 == VF_REAL) {
                realpart(&c1) = dd1[i];
                imagpart(&c1) = 0.0;
            } else {
                realpart(&c1) = realpart(&cc1[i]);
                imagpart(&c1) = imagpart(&cc1[i]);
            }
            if (datatype2 == VF_REAL) {
                realpart(&c2) = dd2[i];
                imagpart(&c2) = 0.0;
            } else {
                realpart(&c2) = realpart(&cc2[i]);
                imagpart(&c2) = imagpart(&cc2[i]);
            }
        rcheck((realpart(&c2) != 0) || (imagpart(&c2) != 0), "divide");
#define xx5 realpart(&c1)
#define xx6 imagpart(&c1)
cdiv(xx5, xx6, realpart(&c2), imagpart(&c2), realpart(&c[i]), imagpart(&c[i]));
        }
        return ((void *) c);
    }
}

/* Should just use "j( )" */
/* The comma operator. What this does (unless it is part of the argument
 * list of a user-defined function) is arg1 + j(arg2).
 */

void *
cx_comma(void *data1, void *data2, short int datatype1, short int datatype2, int length)
{
    double *dd1 = (double *) data1;
    double *dd2 = (double *) data2;
    complex *cc1 = (complex *) data1;
    complex *cc2 = (complex *) data2;
    complex *c, c1, c2;
    int i;

    c = alloc_c(length);
    for (i = 0; i < length; i++) {
        if (datatype1 == VF_REAL) {
            realpart(&c1) = dd1[i];
            imagpart(&c1) = 0.0;
        } else {
            realpart(&c1) = realpart(&cc1[i]);
            imagpart(&c1) = imagpart(&cc1[i]);
        }
        if (datatype2 == VF_REAL) {
            realpart(&c2) = dd2[i];
            imagpart(&c2) = 0.0;
        } else {
            realpart(&c2) = realpart(&cc2[i]);
            imagpart(&c2) = imagpart(&cc2[i]);
        }

        realpart(&c[i]) = realpart(&c1) + imagpart(&c2);
        imagpart(&c[i]) = imagpart(&c1) + realpart(&c2);
    }
    return ((void *) c);
}

void *
cx_power(void *data1, void *data2, short int datatype1, short int datatype2, int length)
{
    double *dd1 = (double *) data1;
    double *dd2 = (double *) data2;
    double *d;
    complex *cc1 = (complex *) data1;
    complex *cc2 = (complex *) data2;
    complex *c, c1, c2, *t;
    int i;

    if ((datatype1 == VF_REAL) && (datatype2 == VF_REAL)) {
        d = alloc_d(length);
        for (i = 0; i < length; i++) {
    rcheck((dd1[i] >= 0) || (floor(dd2[i]) == ceil(dd2[i])), "power");
            d[i] = pow(dd1[i], dd2[i]);
        }
        return ((void *) d);
    } else {
        c = alloc_c(length);
        for (i = 0; i < length; i++) {
            if (datatype1 == VF_REAL) {
                realpart(&c1) = dd1[i];
                imagpart(&c1) = 0.0;
            } else {
                realpart(&c1) = realpart(&cc1[i]);
                imagpart(&c1) = imagpart(&cc1[i]);
            }
            if (datatype2 == VF_REAL) {
                realpart(&c2) = dd2[i];
                imagpart(&c2) = 0.0;
            } else {
                realpart(&c2) = realpart(&cc2[i]);
                imagpart(&c2) = imagpart(&cc2[i]);
            }

            if ((realpart(&c1) == 0.0) && (imagpart(&c1) == 0.0)) {
                realpart(&c[i]) = 0.0;
                imagpart(&c[i]) = 0.0;
            } else { /* if ((imagpart(&c1) != 0.0) && 
                        (imagpart(&c2) != 0.0)) */
                t = cexp(ctimes(&c2, cln(&c1)));
                realpart(&c[i]) = realpart(t);
                imagpart(&c[i]) = imagpart(t);
            /*
            } else {
                realpart(&c[i]) = pow(realpart(&c1), 
                                realpart(&c2)); 
                imagpart(&c[i]) = 0.0;
            */
            }
        }
        return ((void *) c);
    }
}

/* These are unnecessary... Only cx_power uses them... */

static complex *
cexp(complex *c)
{
    static complex r;
    double d;

    d = exp(realpart(c));
    realpart(&r) = d * cos(imagpart(c));
    if (imagpart(c) != 0.0)
        imagpart(&r) = d * sin(imagpart(c));
    else
        imagpart(&r) = 0.0;
    return (&r);
}

static complex *
cln(complex *c)
{
    static complex r;

    rcheck(cmag(c) != 0, "ln");
    realpart(&r) = log(cmag(c));
    if (imagpart(c) != 0.0)
        imagpart(&r) = atan2(imagpart(c), realpart(c));
    else
        imagpart(&r) = 0.0;
    return (&r);
}

static complex *
ctimes(complex *c1, complex *c2)
{
    static complex r;

    realpart(&r) = realpart(c1) * realpart(c2) - 
               imagpart(c1) * imagpart(c2);
    imagpart(&r) = imagpart(c1) * realpart(c2) +
               realpart(c1) * imagpart(c2);
    return (&r);
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
    complex *cc1 = (complex *) data1;
    complex *cc2 = (complex *) data2;
    complex c1, c2;
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
                realpart(&c1) = dd1[i];
                imagpart(&c1) = 0.0;
            } else {
                realpart(&c1) = realpart(&cc1[i]);
                imagpart(&c1) = imagpart(&cc1[i]);
            }
            if (datatype2 == VF_REAL) {
                realpart(&c2) = dd2[i];
                imagpart(&c2) = 0.0;
            } else {
                realpart(&c2) = realpart(&cc2[i]);
                imagpart(&c2) = imagpart(&cc2[i]);
            }
            d[i] = ((realpart(&c1) == realpart(&c2)) &&
                (imagpart(&c1) == imagpart(&c2)));
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
    complex *cc1 = (complex *) data1;
    complex *cc2 = (complex *) data2;
    complex c1, c2;
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
                realpart(&c1) = dd1[i];
                imagpart(&c1) = 0.0;
            } else {
                realpart(&c1) = realpart(&cc1[i]);
                imagpart(&c1) = imagpart(&cc1[i]);
            }
            if (datatype2 == VF_REAL) {
                realpart(&c2) = dd2[i];
                imagpart(&c2) = 0.0;
            } else {
                realpart(&c2) = realpart(&cc2[i]);
                imagpart(&c2) = imagpart(&cc2[i]);
            }
            d[i] = ((realpart(&c1) > realpart(&c2)) &&
                (imagpart(&c1) > imagpart(&c2)));
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
    complex *cc1 = (complex *) data1;
    complex *cc2 = (complex *) data2;
    complex c1, c2;
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
                realpart(&c1) = dd1[i];
                imagpart(&c1) = 0.0;
            } else {
                realpart(&c1) = realpart(&cc1[i]);
                imagpart(&c1) = imagpart(&cc1[i]);
            }
            if (datatype2 == VF_REAL) {
                realpart(&c2) = dd2[i];
                imagpart(&c2) = 0.0;
            } else {
                realpart(&c2) = realpart(&cc2[i]);
                imagpart(&c2) = imagpart(&cc2[i]);
            }
            d[i] = ((realpart(&c1) < realpart(&c2)) &&
                (imagpart(&c1) < imagpart(&c2)));
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
    complex *cc1 = (complex *) data1;
    complex *cc2 = (complex *) data2;
    complex c1, c2;
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
                realpart(&c1) = dd1[i];
                imagpart(&c1) = 0.0;
            } else {
                realpart(&c1) = realpart(&cc1[i]);
                imagpart(&c1) = imagpart(&cc1[i]);
            }
            if (datatype2 == VF_REAL) {
                realpart(&c2) = dd2[i];
                imagpart(&c2) = 0.0;
            } else {
                realpart(&c2) = realpart(&cc2[i]);
                imagpart(&c2) = imagpart(&cc2[i]);
            }
            d[i] = ((realpart(&c1) >= realpart(&c2)) &&
                (imagpart(&c1) >= imagpart(&c2)));
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
    complex *cc1 = (complex *) data1;
    complex *cc2 = (complex *) data2;
    complex c1, c2;
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
                realpart(&c1) = dd1[i];
                imagpart(&c1) = 0.0;
            } else {
                realpart(&c1) = realpart(&cc1[i]);
                imagpart(&c1) = imagpart(&cc1[i]);
            }
            if (datatype2 == VF_REAL) {
                realpart(&c2) = dd2[i];
                imagpart(&c2) = 0.0;
            } else {
                realpart(&c2) = realpart(&cc2[i]);
                imagpart(&c2) = imagpart(&cc2[i]);
            }
            d[i] = ((realpart(&c1) <= realpart(&c2)) &&
                (imagpart(&c1) <= imagpart(&c2)));
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
    complex *cc1 = (complex *) data1;
    complex *cc2 = (complex *) data2;
    complex c1, c2;
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
                realpart(&c1) = dd1[i];
                imagpart(&c1) = 0.0;
            } else {
                realpart(&c1) = realpart(&cc1[i]);
                imagpart(&c1) = imagpart(&cc1[i]);
            }
            if (datatype2 == VF_REAL) {
                realpart(&c2) = dd2[i];
                imagpart(&c2) = 0.0;
            } else {
                realpart(&c2) = realpart(&cc2[i]);
                imagpart(&c2) = imagpart(&cc2[i]);
            }
            d[i] = ((realpart(&c1) != realpart(&c2)) &&
                (imagpart(&c1) != imagpart(&c2)));
        }
    }
    return ((void *) d);
}

