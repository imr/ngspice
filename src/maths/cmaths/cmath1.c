/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Wayne A. Christopher, U. C. Berkeley CAD Group
**********/

/** \file cmath1.c
    \brief Functions for the control language parser: mag, ph, cph, unwrap, j, real, conj, pos, db, log10, log, exp, sqrt, sin, sinh, cos, coh, tan, tanh, atan, sortorder 

    Routines to do complex mathematical functions. These routines require
    the -lm libraries. We sacrifice a lot of space to be able
    to avoid having to do a seperate call for every vector element,
    but it pays off in time savings.  These routines should never
    allow FPE's to happen.

    Complex functions are called as follows:
     cx_something(data, type, length, &newlength, &newtype),
     and return a void* that has to be cast to complex or double.
     Integers newlength and newtype contain the newly resulting length
     of the void* vector and its new type (REAL or COMPLEX).
*/


#include <errno.h>
#include <complex.h>

#include "ngspice/ngspice.h"
#include "ngspice/memory.h"
#include "ngspice/cpdefs.h"
#include "ngspice/dvec.h"

#include "cmath.h"
#include "cmath1.h"

#ifdef HAS_WINGUI
#define win_x_fprintf fprintf
#endif

/**This global flag determines whether degrees or radians are used. The radtodeg
 * and degtorad macros are no-ops if this is FALSE. It will be set to TRUE in options.c
 * if variable (option) 'unit' is equal to 'degree'. 
 */
bool cx_degrees = FALSE;

/** Magnitude of real and complex vectors:
    fabs() for real,
    hypot() for complex.
*/
void *
cx_mag(void *data, short int type, int length, int *newlength, short int *newtype)
{
    double *d = alloc_d(length);
    double *dd = (double *) data;
    ngcomplex_t *cc = (ngcomplex_t *) data;
    int i;

    *newlength = length;
    *newtype = VF_REAL;
    if (type == VF_REAL)
        for (i = 0; i < length; i++)
            d[i] = fabs(dd[i]);
    else
        for (i = 0; i < length; i++)
            d[i] = cmag(cc[i]);
    return ((void *) d);
}

/** Phase of vectors:
    0 for real,
    atan2() for complex.
*/
void *
cx_ph(void *data, short int type, int length, int *newlength, short int *newtype)
{
    double *d = alloc_d(length);
    ngcomplex_t *cc = (ngcomplex_t *) data;
    int i;

    *newlength = length;
    *newtype = VF_REAL;
    if (type == VF_COMPLEX)
        for (i = 0; i < length; i++) {
            d[i] = radtodeg(cph(cc[i]));
        }
    /* Otherwise it is 0, but tmalloc zeros the stuff already. */
    return ((void *) d);
}

/** Continuous phase of vectors:
    0 for real,
    atan2() for complex.
    Modified from cx_ph to find closest from +2pi,0, -2pi. */
void *
cx_cph(void *data, short int type, int length, int *newlength, short int *newtype)
{
    double *d = alloc_d(length);
    ngcomplex_t *cc = (ngcomplex_t *) data;
    int i;

    *newlength = length;
    *newtype = VF_REAL;
    if (type == VF_COMPLEX) {
        double last_ph = cph(cc[0]);
        d[0] = radtodeg(last_ph);
        for (i = 1; i < length; i++) {
            double ph = cph(cc[i]);
            last_ph = ph - (2*M_PI) * floor((ph - last_ph)/(2*M_PI) + 0.5);
            d[i] = radtodeg(last_ph);
        }
    }
    /* Otherwise it is 0, but tmalloc zeros the stuff already. */
    return ((void *) d);
}

/** Modified from cx_cph(), but with real phase vector in degrees as input.
    Currently not in use.
*/
void *
cx_unwrap(void *data, short int type, int length, int *newlength, short int *newtype)
{
    double *d = alloc_d(length);
    double *dd = (double *) data;
    int i;

    *newlength = length;
    *newtype = VF_REAL;
    if (type == VF_REAL) {
        double last_ph = degtorad(dd[0]);
        d[0] = last_ph;
        for (i = 1; i < length; i++) {
            double ph = degtorad(dd[i]);
            last_ph = ph - (2*M_PI) * floor((ph - last_ph)/(2*M_PI) + 0.5);
            d[i] = radtodeg(last_ph);
        }
    }
    /* Otherwise it is 0, but tmalloc zeros the stuff already. */
    return ((void *) d);
}

/** Multiply by i (imaginary unit). */
void *cx_j(void *data, short int type, int length, int *newlength,
        short int *newtype)
{
    ngcomplex_t *c = alloc_c(length);
    *newlength = length;
    *newtype = VF_COMPLEX;

    if (type == VF_COMPLEX) {
        ngcomplex_t *cc = (ngcomplex_t *) data;
        int i;
        for (i = 0; i < length; i++) {
            realpart(c[i]) = -imagpart(cc[i]);
            imagpart(c[i]) = realpart(cc[i]);
        }
    }
    else {
        double *dd = (double *) data;
        int i;
        for (i = 0; i < length; i++) {
            imagpart(c[i]) = dd[i];
            /* Real part is already 0. */
        }
    }
    return (void *) c;
}

/** Return the real part of the vector. */
void *cx_real(void *data, short int type, int length, int *newlength,
        short int *newtype)
{
    double *d = alloc_d(length);

    *newlength = length;
    *newtype = VF_REAL;
    if (type == VF_COMPLEX) {
        ngcomplex_t *cc = (ngcomplex_t *) data;
        int i;
        for (i = 0; i < length; i++) {
            d[i] = realpart(cc[i]);
        }
    }
    else {
        double *dd = (double *) data;
        int i;
        for (i = 0; i < length; i++) {
            d[i] = dd[i];
        }
    }
    return (void *) d;
}

/** Return the imaginary part of the vector. */
void *cx_imag(void *data, short int type, int length, int *newlength,
        short int *newtype)
{
    double *d = alloc_d(length);

    *newlength = length;
    *newtype = VF_REAL;
    if (type == VF_COMPLEX) {
        ngcomplex_t *cc = (ngcomplex_t *) data;
        int i;
        for (i = 0; i < length; i++) {
            d[i] = imagpart(cc[i]);
        }
    }
    else {
        double *dd = (double *) data;
        int i;
        for (i = 0; i < length; i++) {
            d[i] = dd[i];
        }
    }
    return (void *) d;
}



/** Create complex conjugate of data. */
void *cx_conj(void *data, short int type, int length,
        int *p_newlength, short int *p_newtype)
{
    /* Length and type do not change */
    *p_newlength = length;
    *p_newtype = type;

    /* For complex, copy with conjugation */
    if (type == VF_COMPLEX) {
        ngcomplex_t * const c_dst = alloc_c(length);
        ngcomplex_t *c_dst_cur = c_dst;
        ngcomplex_t *c_src_cur = (ngcomplex_t *) data;
        ngcomplex_t * const c_src_end = c_src_cur + length;
        for ( ; c_src_cur < c_src_end;  c_src_cur++, c_dst_cur++) {
            c_dst_cur->cx_real = c_src_cur->cx_real;
            c_dst_cur->cx_imag = -c_src_cur->cx_imag;
        }
        return (void *) c_dst;
    }

    /* Else real, so just copy */
    return memcpy(alloc_d(length), data, (unsigned int) length * sizeof(double));
} /* end of function cx_conj */



/* Return a vector with 1s for positive and 0 for negative element values of the input vector.
   Currently not in use.
   */
void *
cx_pos(void *data, short int type, int length, int *newlength, short int *newtype)
{
    double *d = alloc_d(length);
    double *dd = (double *) data;
    ngcomplex_t *cc = (ngcomplex_t *) data;
    int i;

    *newlength = length;
    *newtype = VF_REAL;
    if (type == VF_COMPLEX)
        for (i = 0; i < length; i++)
            d[i] = ((realpart(cc[i]) > 0.0) ? 1.0 : 0.0);
    else
        for (i = 0; i < length; i++)
            d[i] = ((dd[i] > 0.0) ? 1.0 : 0.0);
    return ((void *) d);
}

/** Calculatue values in db as 20.0 * log10.
    Prior to this use macro rcheck() to check for input values being positive.
    Return NULL if not.
*/
void *cx_db(void *data, short int type, int length,
        int *newlength, short int *newtype)
{
    int xrc = 0;
    double *d = alloc_d(length);

    *newlength = length;
    *newtype = VF_REAL;
    if (type == VF_COMPLEX) {
        ngcomplex_t *cc = (ngcomplex_t *) data;
        int i;
        for (i = 0; i < length; i++) {
            const double tt = cmag(cc[i]);
            rcheck(tt > 0, "db");
            /*
                if (tt == 0.0)
                    d[i] = 20.0 * - log(HUGE);
                else
            */
            d[i] = 20.0 * log10(tt);
        }
    }
    else {
        double *dd = (double *) data;
        int i;
        for (i = 0; i < length; i++) {
            const double tt = dd[i];
            rcheck(tt > 0, "db");
            /*
                if (dd[i] == 0.0)
                    d[i] = 20.0 * - log(HUGE);
                else
            */
            d[i] = 20.0 * log10(tt);
        }
    }

EXITPOINT:
    if (xrc != 0) {
        txfree(d);
        d = (double *) NULL;
    }
    return ((void *) d);
} /* end of function cx_db */


/** Return the common logarithm.
    Prior to this use macro rcheck() to check for input values being positive or 0.
    Return -log10(HUGE) when magnitude is 0.
    Return NULL if negative.
*/
void *cx_log10(void *data, short int type, int length,
        int *newlength, short int *newtype)
{
    int xrc = 0;
    void *rv;

    if (type == VF_COMPLEX) {
        ngcomplex_t *c;
        ngcomplex_t *cc = (ngcomplex_t *) data;
        int i;

        rv = c = alloc_c(length);
        *newtype = VF_COMPLEX;
        for (i = 0; i < length; i++) {
            double td;

            td = cmag(cc[i]);
            /* Perhaps we should trap when td = 0.0, but Ken wants
             * this to be possible...
             */
            rcheck(td >= 0, "log10");
            if (td == 0.0) {
                realpart(c[i]) = - log10(HUGE);
                imagpart(c[i]) = 0.0;
            }
            else {
                realpart(c[i]) = log10(td);
                imagpart(c[i]) = atan2(imagpart(cc[i]), realpart(cc[i]));
            }
        }
    }
    else {
        double *d;
        double *dd = (double *) data;
        int i;

        rv = d = alloc_d(length);
        *newtype = VF_REAL;
        for (i = 0; i < length; i++) {
            rcheck(dd[i] >= 0, "log10");
            if (dd[i] == 0.0) {
                d[i] = - log10(HUGE);
            }
            else {
                d[i] = log10(dd[i]);
            }
        }
    }

    *newlength = length;

EXITPOINT:
    if (xrc != 0) { /* Free resources on error */
        txfree(rv);
        rv = NULL;
    }

    return rv;
} /* end of function cx_log10 */


/** Return the natural logarithm.
    Prior to this use macro rcheck() to check for input values being positive or 0.
    Return -log(HUGE) when magnitude is 0.
    Return NULL if negative.
*/
void *cx_log(void *data, short int type, int length,
        int *newlength, short int *newtype)
{
    int xrc = 0;
    void *rv;

    if (type == VF_COMPLEX) {
        ngcomplex_t *c;
        ngcomplex_t *cc = (ngcomplex_t *) data;

        rv = c = alloc_c(length);
        *newtype = VF_COMPLEX;

        int i;
        for (i = 0; i < length; i++) {
            double td;

            td = cmag(cc[i]);
            rcheck(td >= 0, "log");
            if (td == 0.0) {
                realpart(c[i]) = - log(HUGE);
                imagpart(c[i]) = 0.0;
            }
            else {
                realpart(c[i]) = log(td);
                imagpart(c[i]) = atan2(imagpart(cc[i]), realpart(cc[i]));
            }
        }
    }
    else {
        double *d;
        double *dd = (double *) data;

        rv = d = alloc_d(length);
        *newtype = VF_REAL;

        int i;
        for (i = 0; i < length; i++) {
            rcheck(dd[i] >= 0, "log");
            if (dd[i] == 0.0)
                d[i] = - log(HUGE);
            else
                d[i] = log(dd[i]);
        }
    }

    *newlength = length;

EXITPOINT:
    if (xrc != 0) { /* Free resources on error */
        txfree(rv);
        rv = NULL;
    }

    return rv;
} /* end of function cx_log */


/** Return the exponential of a vector:
    exp() for real,
    exp(realpart)*cos(imagpart), exp(realpart)*sin(imagpart) for imaginary.
    */
void *
cx_exp(void *data, short int type, int length, int *newlength, short int *newtype)
{
    *newlength = length;
    if (type == VF_COMPLEX) {
        ngcomplex_t *c;
        ngcomplex_t *cc = (ngcomplex_t *) data;
        int i;

        c = alloc_c(length);
        *newtype = VF_COMPLEX;
        for (i = 0; i < length; i++) {
            double td;

            td = exp(realpart(cc[i]));
            realpart(c[i]) = td * cos(imagpart(cc[i]));
            imagpart(c[i]) = td * sin(imagpart(cc[i]));
        }
        return ((void *) c);
    } else {
        double *d;
        double *dd = (double *) data;
        int i;

        d = alloc_d(length);
        *newtype = VF_REAL;
        for (i = 0; i < length; i++)
            d[i] = exp(dd[i]);
        return ((void *) d);
    }
}

/** Square root of a complex vector:
    Determine if the result vector is real or complex (due to input being negative or already complex).
    Distinction of cases: Input complex, then real part pos., neg. or 0. Input real negative or real positive.
    */
void *
cx_sqrt(void *data, short int type, int length, int *newlength, short int *newtype)
{
    double *d = NULL;
    ngcomplex_t *c = NULL;
    double *dd = (double *) data;
    ngcomplex_t *cc = (ngcomplex_t *) data;
    int i, cres = (type == VF_REAL) ? 0 : 1;

    if (type == VF_REAL)
        for (i = 0; i < length; i++)
            if (dd[i] < 0.0)
                cres = 1;
    if (cres) {
        c = alloc_c(length);
        *newtype = VF_COMPLEX;
    } else {
        d = alloc_d(length);
        *newtype = VF_REAL;
    }
    *newlength = length;
    if (type == VF_COMPLEX) {
        for (i = 0; i < length; i++) {
            if (realpart(cc[i]) == 0.0) {
                if (imagpart(cc[i]) == 0.0) {
                    realpart(c[i]) = 0.0;
                    imagpart(c[i]) = 0.0;
                } else if (imagpart(cc[i]) > 0.0) {
                    realpart(c[i]) = sqrt (0.5 *
                                            imagpart(cc[i]));
                    imagpart(c[i]) = realpart(c[i]);
                } else {
                    imagpart(c[i]) = sqrt( -0.5 *
                                            imagpart(cc[i]));
                    realpart(c[i]) = - imagpart(c[i]);
                }
            } else if (realpart(cc[i]) > 0.0) {
                if (imagpart(cc[i]) == 0.0) {
                    realpart(c[i]) =
                        sqrt(realpart(cc[i]));
                    imagpart(c[i]) = 0.0;
                } else if (imagpart(cc[i]) < 0.0) {
                    realpart(c[i]) = -sqrt(0.5 *
                                            (cmag(cc[i]) + realpart(cc[i])));
                } else {
                    realpart(c[i]) = sqrt(0.5 *
                                           (cmag(cc[i]) + realpart(cc[i])));
                }
                imagpart(c[i]) = imagpart(cc[i]) / (2.0 *
                                                      realpart(c[i]));
            } else { /* realpart(cc[i]) < 0.0) */
                if (imagpart(cc[i]) == 0.0) {
                    realpart(c[i]) = 0.0;
                    imagpart(c[i]) =
                        sqrt(- realpart(cc[i]));
                } else {
                    if (imagpart(cc[i]) < 0.0)
                        imagpart(c[i]) = - sqrt(0.5 *
                                                 (cmag(cc[i]) -
                                                  realpart(cc[i])));
                    else
                        imagpart(c[i]) = sqrt(0.5 *
                                               (cmag(cc[i]) -
                                                realpart(cc[i])));
                    realpart(c[i]) = imagpart(cc[i]) /
                                      (2.0 * imagpart(c[i]));
                }
            }
        }
        return ((void *) c);
    } else if (cres) {
        for (i = 0; i < length; i++)
            if (dd[i] < 0.0)
                imagpart(c[i]) = sqrt(- dd[i]);
            else
                realpart(c[i]) = sqrt(dd[i]);
        return ((void *) c);
    } else {
        for (i = 0; i < length; i++)
            d[i] = sqrt(dd[i]);
        return ((void *) d);
    }
}

/** sin of a complex vector: 
    sin(realpart)*cosh(imagpart), cos(realpart)*sinh(imagpart)
    */
void *
cx_sin(void *data, short int type, int length, int *newlength, short int *newtype)
{
    *newlength = length;
    if (type == VF_COMPLEX) {
        ngcomplex_t *c;
        ngcomplex_t *cc = (ngcomplex_t *) data;
        int i;

        c = alloc_c(length);
        *newtype = VF_COMPLEX;
        for (i = 0; i < length; i++) {
            realpart(c[i]) = sin(degtorad(realpart(cc[i]))) *
                              cosh(degtorad(imagpart(cc[i])));
            imagpart(c[i]) = cos(degtorad(realpart(cc[i]))) *
                              sinh(degtorad(imagpart(cc[i])));
        }
        return ((void *) c);
    } else {
        double *d;
        double *dd = (double *) data;
        int i;

        d = alloc_d(length);
        *newtype = VF_REAL;
        for (i = 0; i < length; i++)
            d[i] = sin(degtorad(dd[i]));
        return ((void *) d);
    }
}

/** sinh of a complex vector:
    sinh(x+iy) = sinh(x)*cos(y) + i * cosh(x)*sin(y) */
void *
cx_sinh(void *data, short int type, int length, int *newlength, short int *newtype)
{
    *newlength = length;
    if (type == VF_COMPLEX) {
        ngcomplex_t *c;
        ngcomplex_t *cc = (ngcomplex_t *) data;
        int i;
        double u, v;
        c = alloc_c(length);

        *newtype = VF_COMPLEX;
        for (i = 0; i < length; i++) {
            /* sinh(x+iy) = sinh(x)*cos(y) + i * cosh(x)*sin(y) */
            u = degtorad(realpart(cc[i]));
            v = degtorad(imagpart(cc[i]));
            realpart(c[i]) = sinh(u)*cos(v);
            imagpart(c[i]) = cosh(u)*sin(v);
        }
        return ((void *) c);
    } else {
        double *d;
        double *dd = (double *) data;
        int i;

        d = alloc_d(length);
        *newtype = VF_REAL;
        for (i = 0; i < length; i++)
            d[i] = sinh(degtorad(dd[i]));
        return ((void *) d);
    }
}

/** cos of a complex vector:
    cos(realpart)*cosh(imagpart), -sin(realpart)*sinh(imagpart)
    */
void *
cx_cos(void *data, short int type, int length, int *newlength, short int *newtype)
{
    *newlength = length;
    if (type == VF_COMPLEX) {
        ngcomplex_t *c;
        ngcomplex_t *cc = (ngcomplex_t *) data;
        int i;

        c = alloc_c(length);
        *newtype = VF_COMPLEX;
        for (i = 0; i < length; i++) {
            realpart(c[i]) = cos(degtorad(realpart(cc[i]))) *
                              cosh(degtorad(imagpart(cc[i])));
            imagpart(c[i]) = - sin(degtorad(realpart(cc[i]))) *
                              sinh(degtorad(imagpart(cc[i])));
        }
        return ((void *) c);
    } else {
        double *d;
        double *dd = (double *) data;
        int i;

        d = alloc_d(length);
        *newtype = VF_REAL;
        for (i = 0; i < length; i++)
            d[i] = cos(degtorad(dd[i]));
        return ((void *) d);
    }
}

/**cosh of a complex vector:
   cosh(x+iy) = cosh(x)*cos(y) + i * sinh(x)*sin(y)
   */
void *
cx_cosh(void *data, short int type, int length, int *newlength, short int *newtype)
{
    *newlength = length;
    if (type == VF_COMPLEX) {
        ngcomplex_t *c;
        ngcomplex_t *cc = (ngcomplex_t *) data;
        int i;
        double u, v;

        c = alloc_c(length);

        *newtype = VF_COMPLEX;
        for (i = 0; i < length; i++) {
            /* cosh(x+iy) = cosh(x)*cos(y) + i * sinh(x)*sin(y) */
            u = degtorad(realpart(cc[i]));
            v = degtorad(imagpart(cc[i]));
            realpart(c[i]) = cosh(u)*cos(v);
            imagpart(c[i]) = sinh(u)*sin(v);
        }
        return ((void *) c);
    } else {
        double *d;
        double *dd = (double *) data;
        int i;

        d = alloc_d(length);
        *newtype = VF_REAL;
        for (i = 0; i < length; i++)
            d[i] = cosh(degtorad(dd[i]));
        return ((void *) d);
    }
}


/** tan for real valued vectors. Used by cx_tanh.
    Prior to this use macro rcheck() to check for input values not being 0.
    Return NULL if 0.*/
static double *d_tan(double *dd, int length)
{
    int xrc = 0;
    double *d = alloc_d(length);

    int i;
    for (i = 0; i < length; i++) {
        rcheck(tan(degtorad(dd[i])) != 0, "tan");
        d[i] = tan(degtorad(dd[i]));
    }

EXITPOINT:
    if (xrc != 0) { /* Free resources on error */
        txfree(d);
        d = (double *) NULL;
    }

    return d;
} /* end of function d_tan */


/** tanh for real valued vectors. Used by cx_tanh. */
static double *
d_tanh(double *dd, int length)
{
    double *d;
    int i;

    d = alloc_d(length);
    for (i = 0; i < length; i++) {
        d[i] = tanh(degtorad(dd[i]));
    }
    return d;
}

/** tan of a complex vector. 
 * Used by cx_tan
 * See https://proofwiki.org/wiki/Tangent_of_Complex_Number (formulation 4) among
 * others for the tangent formula:

 * sin z = sin(x + iy) = sin x cos(iy) + cos x sin(iy) = sin x cosh y + i cos x sinh y
 * cos z = cos(x + iy) = cos x cos(iy) + sin x sin(iy) = cos x cosh y - i sin x sinh y
 * tan z = ((sin x cosh y + i cos x sinh y) / (cos x cosh y - i sin x sinh y)) *
            (cos x cosh y + isin x sinh y) / (cos x cosh y + i sin x sinh y)
      = ...
 *
 *
 * tan(a + bi) = (sin(2a) + i * sinh(2b)) / (cos(2a) + cosh(2b))
 */
static ngcomplex_t *c_tan(ngcomplex_t *cc, int length)
{
    ngcomplex_t * const c = alloc_c(length);

    int i;
    for (i = 0; i < length; i++) {
        errno = 0;
        ngcomplex_t *p_dst = c + i;
        ngcomplex_t *p_src = cc + i;
        const double a = p_src->cx_real;
        const double b = p_src->cx_imag;
        const double u = 2 * degtorad(a);
        const double v = 2 * degtorad(b);
        const double n_r = sin(u);
        const double n_i = sinh(v);
        const double d1 = cos(u);
        const double d2 = cosh(v);
        const double d = d1 + d2;
        if (errno != 0 || d == 0.0) {
            (void) fprintf(cp_err,
                    "Invalid argument %lf + %lf i for compex tangent", a, b);
            txfree(c);
            return (ngcomplex_t *) NULL;
        }
        p_dst->cx_real = n_r / d;
        p_dst->cx_imag = n_i / d;
    } /* end of loop over elements in array */
    return c;
} /* end of function c_tan */



/**complex tanh function:
   uses tanh(z) = -i * tan(i * z).
   Used by cx_tanh.
 */
static ngcomplex_t *c_tanh(ngcomplex_t *cc, int length)
{
    ngcomplex_t * const tmp = alloc_c(length); /* i * z */

    /* Build the i * z array to allow tan() to be called */
    {
        int i;
        for (i = 0; i < length; ++i) {
            ngcomplex_t *p_dst = tmp + i;
            ngcomplex_t *p_src = cc + i;

            /* multiply by i */
            p_dst->cx_real = -p_src->cx_imag;
            p_dst->cx_imag = p_src->cx_real;
        }
    }

   /* Calculat tan(i * z), exiting on failure */
    ngcomplex_t *const c = c_tan(tmp, length);
    if (c == (ngcomplex_t *) NULL) {
        txfree(tmp);
        return (ngcomplex_t *) NULL;
    }

    /* Multiply by -i to find final result */
    {
        int i;
        for (i = 0; i < length; ++i) {
            ngcomplex_t *p_cur = c + i;
            const double cx_real = p_cur->cx_real;
            p_cur->cx_real = p_cur->cx_imag;
            p_cur->cx_imag = -cx_real;
        }
    }

    return c;
} /* end of function c_tanh */


/** tan of a complex vector */
void *
cx_tan(void *data, short int type, int length, int *newlength, short int *newtype)
{
    *newlength = length;
    if (type == VF_REAL) {
        *newtype = VF_REAL;
        return (void *) d_tan((double *) data, length);
    } else {
        *newtype = VF_COMPLEX;
        return (void *) c_tan((ngcomplex_t *) data, length);
    }
}

/** tanh of a complex vector */
void *
cx_tanh(void *data, short int type, int length, int *newlength, short int *newtype)
{
    *newlength = length;
    if (type == VF_REAL) {
        *newtype = VF_REAL;
        return (void *) d_tanh((double *) data, length);
    } else {
        *newtype = VF_COMPLEX;
        return (void *) c_tanh((ngcomplex_t *) data, length);
    }
}

/** atanh of a complex vector: use C99 function catanh. */
void*
cx_atanh(void* data, short int type, int length, int* newlength, short int* newtype)
{
    if (type == VF_COMPLEX) {
        ngcomplex_t* d = alloc_c(length);
        *newtype = VF_COMPLEX;
        *newlength = length;
        ngcomplex_t* cc = (ngcomplex_t*)data;
        int i;
        for (i = 0; i < length; i++) {
#ifdef _MSC_VER
            _Dcomplex midin = _Cbuild(degtorad(realpart(cc[i])), degtorad(imagpart(cc[i])));
            _Dcomplex midout = catanh(midin);
#else
            double complex midin = degtorad(realpart(cc[i])) + _Complex_I * degtorad(imagpart(cc[i]));
            double complex midout = catanh(midin);
#endif
            d[i].cx_real = creal(midout);
            d[i].cx_imag = cimag(midout);
        }
        return ((void*)d);
    }
    else {
        double* d = alloc_d(length);
        *newtype = VF_REAL;
        *newlength = length;
        double* cc = (double*)data;
        int i;
        for (i = 0; i < length; i++) {
            d[i] = atanh(cc[i]);
        }
        return ((void*)d);
    }
}

/** atan of a complex vector: return atan of the real part. */
void *
cx_atan(void *data, short int type, int length, int *newlength, short int *newtype)
{
    double *d;

    d = alloc_d(length);
    *newtype = VF_REAL;
    *newlength = length;
    if (type == VF_COMPLEX) {
        ngcomplex_t *cc = (ngcomplex_t *) data;
        int i;

        for (i = 0; i < length; i++)
            d[i] = radtodeg(atan(realpart(cc[i])));
    } else {
        double *dd = (double *) data;
        int i;

        for (i = 0; i < length; i++)
            d[i] = radtodeg(atan(dd[i]));
    }
    return ((void *) d);
}

/** Struct to store and order the values of the amplitudes preserving the index in the original array */
typedef struct {
    double amplitude;
    int index;
} amplitude_index_t;

static int compare_structs (const void *a, const void *b);

/**
 *  Returns the positions of the elements in a real vector
 *  after they have been sorted into increasing order using a stable method (qsort).
 */
void *
cx_sortorder(void *data, short int type, int length, int *newlength, short int *newtype)
{
    double *d = alloc_d(length);
    double *dd = (double *) data;
    int i;

    amplitude_index_t * const array_amplitudes = (amplitude_index_t *)
            tmalloc(sizeof(amplitude_index_t) * (size_t) length);

    *newlength = length;
    *newtype = VF_REAL;
    if (type == VF_REAL) {

        for(i = 0; i < length; i++){
            array_amplitudes[i].amplitude = dd[i];
            array_amplitudes[i].index = i;
        }

        qsort(array_amplitudes, (size_t) length, sizeof(array_amplitudes[0]), compare_structs);

        for(i = 0; i < length; i++)
            d[i] = array_amplitudes[i].index;
    }

    txfree(array_amplitudes);

    /* Otherwise it is 0, but tmalloc zeros the stuff already. */
    return ((void *) d);
}

/** Compares ampplitudes of vector elements. Input to qsort. */
static int
compare_structs(const void *a, const void *b)
{
    amplitude_index_t *aa = (amplitude_index_t *) a;
    amplitude_index_t *bb = (amplitude_index_t *) b;

    if (aa->amplitude > bb->amplitude)
        return 1;
    else if (aa->amplitude == bb->amplitude)
        return 0;
    else
        return -1;
}
