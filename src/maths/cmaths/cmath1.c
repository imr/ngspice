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
 *
 */

#include "ngspice/ngspice.h"
#include "ngspice/memory.h"
#include "ngspice/cpdefs.h"
#include "ngspice/dvec.h"

#include "cmath.h"
#include "cmath1.h"

#ifdef HAS_WINGUI
#define win_x_fprintf fprintf
#endif

/* This flag determines whether degrees or radians are used. The radtodeg
 * and degtorad macros are no-ops if this is FALSE.
 */

bool cx_degrees = FALSE;

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

/* SJdV Modified from above to find closest from +2pi,0, -2pi */
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

/* Modified from above but with real phase vector in degrees as input */
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

/* If this is pure imaginary we might get real, but never mind... */

void *
cx_j(void *data, short int type, int length, int *newlength, short int *newtype)
{
    ngcomplex_t *c = alloc_c(length);
    ngcomplex_t *cc = (ngcomplex_t *) data;
    double *dd = (double *) data;
    int i;

    *newlength = length;
    *newtype = VF_COMPLEX;
    if (type == VF_COMPLEX)
        for (i = 0; i < length; i++) {
            realpart(c[i]) = - imagpart(cc[i]);
            imagpart(c[i]) = realpart(cc[i]);
        }
    else
        for (i = 0; i < length; i++) {
            imagpart(c[i]) = dd[i];
            /* Real part is already 0. */
        }
    return ((void *) c);
}

void *
cx_real(void *data, short int type, int length, int *newlength, short int *newtype)
{
    double *d = alloc_d(length);
    double *dd = (double *) data;
    ngcomplex_t *cc = (ngcomplex_t *) data;
    int i;

    *newlength = length;
    *newtype = VF_REAL;
    if (type == VF_COMPLEX)
        for (i = 0; i < length; i++)
            d[i] = realpart(cc[i]);
    else
        for (i = 0; i < length; i++)
            d[i] = dd[i];
    return ((void *) d);
}

void *
cx_imag(void *data, short int type, int length, int *newlength, short int *newtype)
{
    double *d = alloc_d(length);
    double *dd = (double *) data;
    ngcomplex_t *cc = (ngcomplex_t *) data;
    int i;

    *newlength = length;
    *newtype = VF_REAL;
    if (type == VF_COMPLEX)
        for (i = 0; i < length; i++)
            d[i] = imagpart(cc[i]);
    else
        for (i = 0; i < length; i++)
            d[i] = dd[i];
    return ((void *) d);
}

/* This is obsolete... */

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

void *
cx_db(void *data, short int type, int length, int *newlength, short int *newtype)
{
    double *d = alloc_d(length);
    double *dd = (double *) data;
    ngcomplex_t *cc = (ngcomplex_t *) data;
    double tt;
    int i;

    *newlength = length;
    *newtype = VF_REAL;
    if (type == VF_COMPLEX)
        for (i = 0; i < length; i++) {
            tt = cmag(cc[i]);
            rcheck(tt > 0, "db");
            /*
                if (tt == 0.0)
                    d[i] = 20.0 * - log(HUGE);
                else
            */
            d[i] = 20.0 * log10(tt);
        }
    else
        for (i = 0; i < length; i++) {
            rcheck(dd[i] > 0, "db");
            /*
                if (dd[i] == 0.0)
                    d[i] = 20.0 * - log(HUGE);
                else
            */
            d[i] = 20.0 * log10(dd[i]);
        }
    return ((void *) d);
}

void *
cx_log10(void *data, short int type, int length, int *newlength, short int *newtype)
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

            td = cmag(cc[i]);
            /* Perhaps we should trap when td = 0.0, but Ken wants
             * this to be possible...
             */
            rcheck(td >= 0, "log10");
            if (td == 0.0) {
                realpart(c[i]) = - log10(HUGE);
                imagpart(c[i]) = 0.0;
            } else {
                realpart(c[i]) = log10(td);
                imagpart(c[i]) = atan2(imagpart(cc[i]),
                                        realpart(cc[i]));
            }
        }
        return ((void *) c);
    } else {
        double *d;
        double *dd = (double *) data;
        int i;

        d = alloc_d(length);
        *newtype = VF_REAL;
        for (i = 0; i < length; i++) {
            rcheck(dd[i] >= 0, "log10");
            if (dd[i] == 0.0)
                d[i] = - log10(HUGE);
            else
                d[i] = log10(dd[i]);
        }
        return ((void *) d);
    }
}

void *
cx_log(void *data, short int type, int length, int *newlength, short int *newtype)
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

            td = cmag(cc[i]);
            rcheck(td >= 0, "log");
            if (td == 0.0) {
                realpart(c[i]) = - log(HUGE);
                imagpart(c[i]) = 0.0;
            } else {
                realpart(c[i]) = log(td);
                imagpart(c[i]) = atan2(imagpart(cc[i]),
                                        realpart(cc[i]));
            }
        }
        return ((void *) c);
    } else {
        double *d;
        double *dd = (double *) data;
        int i;

        d = alloc_d(length);
        *newtype = VF_REAL;
        for (i = 0; i < length; i++) {
            rcheck(dd[i] >= 0, "log");
            if (dd[i] == 0.0)
                d[i] = - log(HUGE);
            else
                d[i] = log(dd[i]);
        }
        return ((void *) d);
    }
}

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

static double *
d_tan(double *dd, int length)
{
    double *d;
    int i;

    d = alloc_d(length);
    for (i = 0; i < length; i++) {
        rcheck(tan(degtorad(dd[i])) != 0, "tan");
        d[i] = tan(degtorad(dd[i]));
    }
    return d;
}

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

static ngcomplex_t *
c_tan(ngcomplex_t *cc, int length)
{
    ngcomplex_t *c;
    int i;

    c = alloc_c(length);
    for (i = 0; i < length; i++) {
        double u, v;

        rcheck(cos(degtorad(realpart(cc[i]))) *
               cosh(degtorad(imagpart(cc[i]))), "tan");
        rcheck(sin(degtorad(realpart(cc[i]))) *
               sinh(degtorad(imagpart(cc[i]))), "tan");
        u = degtorad(realpart(cc[i]));
        v = degtorad(imagpart(cc[i]));
        /* The Lattice C compiler won't take multi-line macros, and
         * CMS won't take >80 column lines....
         */
#define xx1 sin(u) * cosh(v)
#define xx2 cos(u) * sinh(v)
#define xx3 cos(u) * cosh(v)
#define xx4 -sin(u) * sinh(v)
        cdiv(xx1, xx2, xx3, xx4, realpart(c[i]), imagpart(c[i]));
    }
    return c;
}

/* complex tanh function, uses tanh(z) = -i * tan (i * z) */
static ngcomplex_t *
c_tanh(ngcomplex_t *cc, int length)
{
    ngcomplex_t *c, *s, *t;
    int i;

    c = alloc_c(length);
    s = alloc_c(1);
    t = alloc_c(1);

    for (i = 0; i < length; i++) {
        /* multiply by i */
        t[0].cx_real = -1. * imagpart(cc[i]);
        t[0].cx_imag = realpart(cc[i]);
        /* get complex tangent */
        s = c_tan(t, 1);
        /* if check in c_tan fails */
        if (s == NULL) {
            tfree(t);
            return (NULL);
        }
        /* multiply by -i */
        realpart(c[i]) = imagpart(s[0]);
        imagpart(c[i]) = -1. * realpart(s[0]);
    }
    tfree(s);
    tfree(t);
    return c;
}

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

/* Struct to store and order the values of the amplitudes preserving the index in the original array */
typedef struct {
    double amplitude;
    int index;
} amplitude_index_t;

static int compare_structs (const void *a, const void *b);

/*
 *  Returns the positions of the elements in a real vector
 *  after they have been sorted into increasing order using a stable method (qsort).
 */

void *
cx_sortorder(void *data, short int type, int length, int *newlength, short int *newtype)
{
    double *d = alloc_d(length);
    double *dd = (double *) data;
    int i;

    amplitude_index_t *array_amplitudes;
    array_amplitudes = (amplitude_index_t *) tmalloc(sizeof(amplitude_index_t) * (size_t) length);

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

    tfree(array_amplitudes);

    /* Otherwise it is 0, but tmalloc zeros the stuff already. */
    return ((void *) d);
}

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
