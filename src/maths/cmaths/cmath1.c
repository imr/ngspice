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

#include <ngspice.h>
#include <memory.h>
#include <cpdefs.h>
#include <dvec.h>

#include "cmath.h"
#include "cmath1.h"

#ifdef HAS_WINDOWS
#define fp_r_i_n_t_f fprintf
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
    complex *cc = (complex *) data;
    int i;

    *newlength = length;
    *newtype = VF_REAL;
    if (type == VF_REAL)
        for (i = 0; i < length; i++)
            d[i] = FTEcabs(dd[i]);
    else
        for (i = 0; i < length; i++)
            d[i] = cmag(&cc[i]);
    return ((void *) d);
}

void *
cx_ph(void *data, short int type, int length, int *newlength, short int *newtype)
{
    double *d = alloc_d(length);
    complex *cc = (complex *) data;
    int i;

    *newlength = length;
    *newtype = VF_REAL;
    if (type == VF_COMPLEX)
        for (i = 0; i < length; i++) {
            d[i] = radtodeg(cph(&cc[i]));
        }
    /* Otherwise it is 0, but tmalloc zeros the stuff already. */
    return ((void *) d);
}

/* If this is pure imaginary we might get real, but never mind... */

void *
cx_j(void *data, short int type, int length, int *newlength, short int *newtype)
{
    complex *c = alloc_c(length);
    complex *cc = (complex *) data;
    double *dd = (double *) data;
    int i;

    *newlength = length;
    *newtype = VF_COMPLEX;
    if (type == VF_COMPLEX)
        for (i = 0; i < length; i++) {
            realpart(&c[i]) = - imagpart(&cc[i]);
            imagpart(&c[i]) = realpart(&cc[i]);
        }
    else
        for (i = 0; i < length; i++) {
            imagpart(&c[i]) = dd[i];
            /* Real part is already 0. */
        }
    return ((void *) c);
}

void *
cx_real(void *data, short int type, int length, int *newlength, short int *newtype)
{
    double *d = alloc_d(length);
    double *dd = (double *) data;
    complex *cc = (complex *) data;
    int i;

    *newlength = length;
    *newtype = VF_REAL;
    if (type == VF_COMPLEX)
        for (i = 0; i < length; i++)
            d[i] = realpart(&cc[i]);
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
    complex *cc = (complex *) data;
    int i;

    *newlength = length;
    *newtype = VF_REAL;
    if (type == VF_COMPLEX)
        for (i = 0; i < length; i++)
            d[i] = imagpart(&cc[i]);
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
    complex *cc = (complex *) data;
    int i;

    *newlength = length;
    *newtype = VF_REAL;
    if (type == VF_COMPLEX)
        for (i = 0; i < length; i++)
            d[i] = ((realpart(&cc[i]) > 0.0) ? 1.0 : 0.0);
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
    complex *cc = (complex *) data;
    double tt;
    int i;

    *newlength = length;
    *newtype = VF_REAL;
    if (type == VF_COMPLEX)
        for (i = 0; i < length; i++) {
            tt = cmag(&cc[i]);
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
cx_log(void *data, short int type, int length, int *newlength, short int *newtype)
{
    *newlength = length;
    if (type == VF_COMPLEX) {
	complex *c;
	complex *cc = (complex *) data;
	int i;

        c = alloc_c(length);
        *newtype = VF_COMPLEX;
        for (i = 0; i < length; i++) {
	    double td;

            td = cmag(&cc[i]);
            /* Perhaps we should trap when td = 0.0, but Ken wants
             * this to be possible...
             */
            rcheck(td >= 0, "log");
            if (td == 0.0) {
                realpart(&c[i]) = - log10(HUGE);
                imagpart(&c[i]) = 0.0;
            } else {
                realpart(&c[i]) = log10(td);
                imagpart(&c[i]) = atan2(imagpart(&cc[i]),
                        realpart(&cc[i]));
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
                d[i] = - log10(HUGE);
            else 
                d[i] = log10(dd[i]);
        }
        return ((void *) d);
    }
}

void *
cx_ln(void *data, short int type, int length, int *newlength, short int *newtype)
{
    *newlength = length;
    if (type == VF_COMPLEX) {
	complex *c;
	complex *cc = (complex *) data;
	int i;

        c = alloc_c(length);
        *newtype = VF_COMPLEX;
        for (i = 0; i < length; i++) {
	    double td;

            td = cmag(&cc[i]);
            rcheck(td >= 0, "ln");
            if (td == 0.0) {
                realpart(&c[i]) = - log(HUGE);
                imagpart(&c[i]) = 0.0;
            } else {
                realpart(&c[i]) = log(td);
                imagpart(&c[i]) = atan2(imagpart(&cc[i]),
                    realpart(&cc[i]));
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
            rcheck(dd[i] >= 0, "ln");
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
	complex *c;
	complex *cc = (complex *) data;
	int i;

        c = alloc_c(length);
        *newtype = VF_COMPLEX;
        for (i = 0; i < length; i++) {
	    double td;

            td = exp(realpart(&cc[i]));
            realpart(&c[i]) = td * cos(imagpart(&cc[i]));
            imagpart(&c[i]) = td * sin(imagpart(&cc[i]));
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
    complex *c = NULL;
    double *dd = (double *) data;
    complex *cc = (complex *) data;
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
            if (realpart(&cc[i]) == 0.0) {
                if (imagpart(&cc[i]) == 0.0) {
                    realpart(&c[i]) = 0.0;
                    imagpart(&c[i]) = 0.0;
                } else if (imagpart(&cc[i]) > 0.0) {
                    realpart(&c[i]) = sqrt (0.5 *
                            imagpart(&cc[i]));
                    imagpart(&c[i]) = realpart(&c[i]);
                } else { 
                    imagpart(&c[i]) = sqrt( -0.5 * 
                            imagpart(&cc[i]));
                    realpart(&c[i]) = - imagpart(&c[i]);
                }
            } else if (realpart(&cc[i]) > 0.0) {
                if (imagpart(&cc[i]) == 0.0) {
                    realpart(&c[i]) = 
                        sqrt(realpart(&cc[i]));
                    imagpart(&c[i]) = 0.0;
                } else if (imagpart(&cc[i]) < 0.0) {
                    realpart(&c[i]) = -sqrt(0.5 * 
                        (cmag(&cc[i]) + realpart(&cc[i])));
                } else {
                    realpart(&c[i]) = sqrt(0.5 * 
                        (cmag(&cc[i]) + realpart(&cc[i])));
                }
                imagpart(&c[i]) = imagpart(&cc[i]) / (2.0 * 
                        realpart(&c[i]));
            } else { /* realpart(&cc[i]) < 0.0) */
                if (imagpart(&cc[i]) == 0.0) {
                    realpart(&c[i]) = 0.0;
                    imagpart(&c[i]) =
                        sqrt(- realpart(&cc[i]));
                } else {
                    if (imagpart(&cc[i]) < 0.0)
                        imagpart(&c[i]) = - sqrt(0.5 *
                            (cmag(&cc[i]) -
                            realpart(&cc[i])));
                    else
                        imagpart(&c[i]) = sqrt(0.5 *
                            (cmag(&cc[i]) -
                            realpart(&cc[i])));
                    realpart(&c[i]) = imagpart(&cc[i]) / 
                        (2.0 * imagpart(&c[i]));
                }
            }
        }
        return ((void *) c);
    } else if (cres) {
        for (i = 0; i < length; i++)
            if (dd[i] < 0.0)
                imagpart(&c[i]) = sqrt(- dd[i]);
            else
                realpart(&c[i]) = sqrt(dd[i]);
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
	complex *c;
	complex *cc = (complex *) data;
	int i;

        c = alloc_c(length);
        *newtype = VF_COMPLEX;
        for (i = 0; i < length; i++) {
            realpart(&c[i]) = sin(degtorad(realpart(&cc[i]))) *
                    cosh(degtorad(imagpart(&cc[i])));
            imagpart(&c[i]) = cos(degtorad(realpart(&cc[i]))) *
                    sinh(degtorad(imagpart(&cc[i])));
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
cx_cos(void *data, short int type, int length, int *newlength, short int *newtype)
{
    *newlength = length;
    if (type == VF_COMPLEX) {
	complex *c;
	complex *cc = (complex *) data;
	int i;

        c = alloc_c(length);
        *newtype = VF_COMPLEX;
        for (i = 0; i < length; i++) {
            realpart(&c[i]) = cos(degtorad(realpart(&cc[i]))) *
                    cosh(degtorad(imagpart(&cc[i])));
            imagpart(&c[i]) = - sin(degtorad(realpart(&cc[i]))) *
                    sinh(degtorad(imagpart(&cc[i])));
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

