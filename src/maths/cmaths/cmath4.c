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
#include <plot.h>
#include <complex.h>
#include <cpdefs.h>

#include <interpolate.h>
#include <polyfit.h>
#include <polyeval.h>
#include <polyderiv.h>

#include "cmath.h"
#include "cmath4.h"

#include "../../frontend/variable.h" /* for VT_NUM in cx_interpolate */

void *
cx_and(void *data1, void *data2, short int datatype1, short int datatype2, int length)
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
            d[i] = dd1[i] && dd2[i];
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
            d[i] = ((realpart(&c1) && realpart(&c2)) &&
                (imagpart(&c1) && imagpart(&c2)));
        }
    }
    return ((void *) d);
}

void *
cx_or(void *data1, void *data2, short int datatype1, short int datatype2, int length)
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
            d[i] = dd1[i] || dd2[i];
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
            d[i] = ((realpart(&c1) || realpart(&c2)) &&
                (imagpart(&c1) || imagpart(&c2)));
        }
    }
    return ((void *) d);
}

void *
cx_not(void *data, short int type, int length, int *newlength, short int *newtype)
{
    double *d;
    double *dd = (double *) data;
    complex *cc = (complex *) data;
    int i;

    d = alloc_d(length);
    *newtype = VF_REAL;
    *newlength = length;
    if (type == VF_COMPLEX) {
        for (i = 0; i < length; i++) {
            /* gcc doens't like !double */
            d[i] = realpart(&cc[i]) ? 0 : 1;
            d[i] = imagpart(&cc[i]) ? 0 : 1;
        }
    } else {
        for (i = 0; i < length; i++)
            d[i] = ! dd[i];
    }
    return ((void *) d);
}



/* This is a strange function. What we do is fit a polynomial to the
 * curve, of degree $polydegree, and then evaluate it at the points
 * in the time scale.  What we do is this: for every set of points that
 * we fit a polynomial to, fill in as much of the new vector as we can
 * (i.e, between the last value of the old scale we went from to this
 * one). At the ends we just use what we have...  We have to detect
 * badness here too...
 *
 * Note that we pass arguments differently for this one cx_ function...  */
void *
cx_interpolate(void *data, short int type, int length, int *newlength, short int *newtype, struct plot *pl, struct plot *newpl, int grouping)
{
    struct dvec *ns, *os;
    double *d;
    int degree;
    register int i, oincreasing = 1, nincreasing = 1;
    int base;

    if (grouping == 0)
	grouping = length;

    /* First do some sanity checks. */
    if (!pl || !pl->pl_scale || !newpl || !newpl->pl_scale) {
        fprintf(cp_err, "Internal error: cx_interpolate: bad scale\n");
        return (NULL);
    }
    ns = newpl->pl_scale;
    os = pl->pl_scale;
    if (iscomplex(ns)) {
        fprintf(cp_err, "Error: new scale has complex data\n");
        return (NULL);
    }
    if (iscomplex(os)) {
        fprintf(cp_err, "Error: old scale has complex data\n");
        return (NULL);
    }

    if (length != os->v_length) {
        fprintf(cp_err, "Error: lengths don't match\n");
        return (NULL);
    }
    if (type != VF_REAL) {
        fprintf(cp_err, "Error: argument has complex data\n");
        return (NULL);
    }

    /* Now make sure that either both scales are strictly increasing
     * or both are strictly decreasing.  */
    if (os->v_realdata[0] < os->v_realdata[1])
        oincreasing = TRUE;
    else
        oincreasing = FALSE;
    for (i = 0; i < os->v_length - 1; i++)
        if ((os->v_realdata[i] < os->v_realdata[i + 1])
                != oincreasing) {
            fprintf(cp_err, "Error: old scale not monotonic\n");
            return (NULL);
        }
    if (ns->v_realdata[0] < ns->v_realdata[1])
        nincreasing = TRUE;
    else
        nincreasing = FALSE;
    for (i = 0; i < ns->v_length - 1; i++)
        if ((ns->v_realdata[i] < ns->v_realdata[i + 1])
                != nincreasing) {
            fprintf(cp_err, "Error: new scale not monotonic\n");
            return (NULL);
        }

    *newtype = VF_REAL;
    *newlength = ns->v_length;
    d = alloc_d(ns->v_length);

    if (!cp_getvar("polydegree", VT_NUM, (void *) &degree))
        degree = 1;

    for (base = 0; base < length; base += grouping) {
	if (!ft_interpolate((double *) data + base, d + base,
	    os->v_realdata + base, grouping,
            ns->v_realdata + base, grouping, degree))
	{
	    tfree(d);
	    return (NULL);
	}
    }

    return ((void *) d);
}

void *
cx_deriv(void *data, short int type, int length, int *newlength, short int *newtype, struct plot *pl, struct plot *newpl, int grouping)
{
    double *scratch;
    double *spare;
    double x;
    int i, j, k;
    int	degree;
    int n, base;

    if (grouping == 0)
	grouping = length;
    /* First do some sanity checks. */
    if (!pl || !pl->pl_scale || !newpl || !newpl->pl_scale) {
        fprintf(cp_err, "Internal error: cx_deriv: bad scale\n");
        return (NULL);
    }

    if (!cp_getvar("dpolydegree", VT_NUM, (void *) &degree))
	degree = 2; /* default quadratic */

    n = degree +  1;

    spare = alloc_d(n);
    scratch = alloc_d(n * (n + 1));

    *newlength = length;
    *newtype = type;

    if (type == VF_COMPLEX) {
	complex *c_outdata, *c_indata;
	double *r_coefs, *i_coefs;
	double *scale;

	r_coefs = alloc_d(n);
	i_coefs = alloc_d(n);
	c_indata = (complex *) data;
	c_outdata = alloc_c(length);
	scale = alloc_d(length);	/* XXX */
	if (pl->pl_scale->v_type == VF_COMPLEX)
	    /* Not ideal */
	    for (i = 0; i < length; i++)
		scale[i] = realpart(&pl->pl_scale->v_compdata[i]);
	else
	    for (i = 0; i < length; i++)
		scale[i] = pl->pl_scale->v_realdata[i];
	for (base = 0; base < length; base += grouping) {
	    k = 0;
	    for (i = degree; i < grouping; i += 1) {

		/* real */
		for (j = 0; j < n; j++)
		    spare[j] = c_indata[j + i + base].cx_real;
		if (!ft_polyfit(scale + i + base - degree,
		    spare, r_coefs, degree, scratch))
		{
		    fprintf(stderr, "ft_polyfit @ %d failed\n", i);
		}
		ft_polyderiv(r_coefs, degree);

		/* for loop gets the beginning part */
		for (j = k; j <= i - degree / 2; j++) {
		    x = scale[j + base];
		    c_outdata[j + base].cx_real =
			ft_peval(x, r_coefs, degree - 1);
		}

		/* imag */
		for (j = 0; j < n; j++)
		    spare[j] = c_indata[j + i + base].cx_imag;
		if (!ft_polyfit(scale + i - degree + base,
		    spare, i_coefs, degree, scratch))
		{
		    fprintf(stderr, "ft_polyfit @ %d failed\n", i);
		}
		ft_polyderiv(i_coefs, degree);

		/* for loop gets the beginning part */
		for (j = k; j <= i - degree / 2; j++) {
		    x = scale[j + base];
		    c_outdata[j + base].cx_imag =
			ft_peval(x, i_coefs, degree - 1);
		}
		k = j;
	    }

	    /* get the tail */
	    for (j = k; j < length; j++) {
		x = scale[j + base];
		/* real */
		c_outdata[j + base].cx_real = ft_peval(x, r_coefs, degree - 1);
		/* imag */
		c_outdata[j + base].cx_imag = ft_peval(x, i_coefs, degree - 1);
	    }
	}

	tfree(r_coefs);
	tfree(i_coefs);
	tfree(scale);
	return (void *) c_outdata;

    } else {
	/* all-real case */
	double *coefs;

	double *outdata, *indata;
	double *scale;

	coefs = alloc_d(n);
	indata = (double *) data;
	outdata = alloc_d(length);
	scale = alloc_d(length);	/* XXX */
	for (i = 0; i < length; i++)
	    scale[i] = pl->pl_scale->v_realdata[i];
	for (base = 0; base < length; base += grouping) {
	    k = 0;
	    for (i = degree; i < grouping; i += 1) {
		if (!ft_polyfit(scale + i - degree + base,
		    indata + i - degree + base, coefs, degree, scratch))
		{
		    fprintf(stderr, "ft_polyfit @ %d failed\n", i + base);
		}
		ft_polyderiv(coefs, degree);

		/* for loop gets the beginning part */
		for (j = k; j <= i - degree / 2; j++) {
		    x = pl->pl_scale->v_realdata[j + base];
		    outdata[j + base] = ft_peval(x, coefs, degree - 1);
		}
		k = j;
	    }

	    for (j = k; j < length; j++) {
		    x = pl->pl_scale->v_realdata[j + base];
		    outdata[j + base] = ft_peval(x, coefs, degree - 1);
	    }
        }


	tfree(coefs);
	tfree(scale);	/* XXX */
	return (void *) outdata;
    }

}
