/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Wayne A. Christopher, U. C. Berkeley CAD Group
**********/

/** \file cmath4.c
    \brief functions for the control language parser: and, or, not, interpolate, deriv, integ, group_delay, fft, ifft

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
#include "ngspice/plot.h"
#include "ngspice/complex.h"
#include "ngspice/cpdefs.h"

#include <interpolate.h>
#include <polyfit.h>
#include <polyeval.h>
#include <polyderiv.h>

#include "cmath.h"
#include "cmath4.h"

#include "ngspice/sim.h" /* To get SV_TIME */
#include "ngspice/fftext.h"

extern bool cx_degrees;
extern void vec_new(struct dvec *d);

#ifdef HAVE_LIBFFTW3
#include "fftw3.h"
#endif


void *
cx_and(void *data1, void *data2, short int datatype1, short int datatype2, int length)
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
            d[i] = dd1[i] && dd2[i];
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
            d[i] = ((realpart(c1) && realpart(c2)) &&
                (imagpart(c1) && imagpart(c2)));
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
    ngcomplex_t *cc1 = (ngcomplex_t *) data1;
    ngcomplex_t *cc2 = (ngcomplex_t *) data2;
    ngcomplex_t c1, c2;
    int i;

    d = alloc_d(length);
    if ((datatype1 == VF_REAL) && (datatype2 == VF_REAL)) {
        for (i = 0; i < length; i++)
            d[i] = dd1[i] || dd2[i];
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
            d[i] = ((realpart(c1) || realpart(c2)) &&
                (imagpart(c1) || imagpart(c2)));
        }
    }
    return ((void *) d);
}


void *
cx_not(void *data, short int type, int length, int *newlength, short int *newtype)
{
    double *d;
    double *dd = (double *) data;
    ngcomplex_t *cc = (ngcomplex_t *) data;
    int i;

    d = alloc_d(length);
    *newtype = VF_REAL;
    *newlength = length;
    if (type == VF_COMPLEX) {
        for (i = 0; i < length; i++) {
            /* gcc doens't like !double */
            d[i] = realpart(cc[i]) ? 0 : 1;
            d[i] = imagpart(cc[i]) ? 0 : 1;
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
 * Note that we pass arguments differently for this one cx_ function...
 */

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

    if (grouping != length) {
        fprintf(cp_err, "Error: interpolation of multi-dimensional vectors is currently not supported\n");
        return (NULL);
    }
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

    if (!cp_getvar("polydegree", CP_NUM, &degree, 0))
        degree = 1;

    /* FIXME: this function is defect: ns data cannot have same base and grouping 
       as the original data. Will now do only if grouping == length. */
    for (base = 0; base < length; base += grouping) {
        if (!ft_interpolate((double *) data + base, d + base,
            os->v_realdata + base, grouping,
            ns->v_realdata + base, ns->v_length, degree))
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
    int degree;
    int n, base;

    if (grouping == 0)
        grouping = length;
    /* First do some sanity checks. */
    if (!pl || !pl->pl_scale || !newpl || !newpl->pl_scale) {
        fprintf(cp_err, "Internal error: cx_deriv: bad scale\n");
        return (NULL);
    }

    if (!cp_getvar("dpolydegree", CP_NUM, &degree, 0))
        degree = 2; /* default quadratic */

    n = degree + 1;

    spare = alloc_d(n);
    scratch = alloc_d(n * (n + 1));

    *newlength = length;
    *newtype = type;

    if (type == VF_COMPLEX) {
        ngcomplex_t *c_outdata, *c_indata;
        double *r_coefs, *i_coefs;
        double *scale;

        r_coefs = alloc_d(n);
        i_coefs = alloc_d(n);
        c_indata = (ngcomplex_t *) data;
        c_outdata = alloc_c(length);
        scale = alloc_d(length);        /* XXX */
        if (iscomplex(pl->pl_scale))
            /* Not ideal */
            for (i = 0; i < length; i++)
                scale[i] = realpart(pl->pl_scale->v_compdata[i]);
        else
            for (i = 0; i < length; i++)
                scale[i] = pl->pl_scale->v_realdata[i];

        for (base = 0; base < length; base += grouping)
        {
            k = 0;
            for (i = degree; i < grouping; i += 1)
            {

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
                for (j = k; j <= i + degree / 2; j++)
                {
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
                for (j = k; j <= i - degree / 2; j++)
                {
                    x = scale[j + base];
                    c_outdata[j + base].cx_imag =
                    ft_peval(x, i_coefs, degree - 1);
                }
                k = j;
            }

            /* get the tail */
            for (j = k; j < length; j++)
            {
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
        tfree(spare);
        tfree(scratch);
        return (void *) c_outdata;

    }
    else
    {
        /* all-real case */
        double *coefs;

        double *outdata, *indata;
        double *scale;

        coefs = alloc_d(n);
        indata = (double *) data;
        outdata = alloc_d(length);
        scale = alloc_d(length);        /* XXX */

        /* Here I encountered a problem because when we issue an instruction like this:
         * plot -deriv(vp(3)) to calculate something similar to the group delay, the code
         * detects that vector vp(3) is real and it is believed that the frequency is also
         * real. The frequency is COMPLEX and the program aborts so I'm going to put the
         * check that the frequency is complex vector not to abort.
         */

        /* Modified to deal with complex frequency vector */
        if (iscomplex(pl->pl_scale))
            for (i = 0; i < length; i++)
                scale[i] = realpart(pl->pl_scale->v_compdata[i]);
        else
            for (i = 0; i < length; i++)
                scale[i] = pl->pl_scale->v_realdata[i];


        for (base = 0; base < length; base += grouping)
        {
            k = 0;
            for (i = degree; i < grouping; i += 1)
            {
                if (!ft_polyfit(scale + i - degree + base,
                    indata + i - degree + base, coefs, degree, scratch))
                {
                    fprintf(stderr, "ft_polyfit @ %d failed\n", i + base);
                }
                ft_polyderiv(coefs, degree);

                /* for loop gets the beginning part */
                for (j = k; j <= i - degree / 2; j++)
                {
                    /* Seems the same problem because the frequency vector is complex
                     * and the real part of the complex should be accessed because if we
                     * run x = pl-> pl_scale-> v_realdata [base + j]; the execution will
                     * abort.
                     */

                    if (iscomplex(pl->pl_scale))
                        x = realpart(pl->pl_scale->v_compdata[j+base]);  /* For complex scale vector */
                    else
                        x = pl->pl_scale->v_realdata[j + base];           /* For real scale vector */

                    outdata[j + base] = ft_peval(x, coefs, degree - 1);
                }
                k = j;
            }

            /* FIXME: replaced j+base by j, to avoid crash, but why j+base here? */
            for (j = k; j < length; j++)
            {
                if (iscomplex(pl->pl_scale))
                    x = realpart(pl->pl_scale->v_compdata[j]);  /* For complex scale vector */
                else
                    x = pl->pl_scale->v_realdata[j];           /* For real scale vector */

                outdata[j] = ft_peval(x, coefs, degree - 1);
            }
        }


        tfree(coefs);
        tfree(scale);
        tfree(spare);
        tfree(scratch);
        return (char *) outdata;
    }

}

/* integrate a vector using trapezoidal rule */
void*
cx_integ(void* data, short int type, int length, int* newlength, short int* newtype, struct plot* pl, struct plot* newpl, int grouping)
{
    if (grouping == 0)
        grouping = length;
    /* First do some sanity checks. */
    if (!pl || !pl->pl_scale || !newpl || !newpl->pl_scale) {
        fprintf(cp_err, "Internal error: cx_integ: bad scale\n");
        return (NULL);
    }

    *newlength = length;
    *newtype = type;

    if (type == VF_COMPLEX) {
        fprintf(cp_err, "Error: Function integ is not supported for complex data\n");
        return (NULL);
    }
    else
    {
        /* all-real case */
        double* outdata, * indata;
        double* scale;
        int i;
        double delta;

        indata = (double*)data;
        outdata = alloc_d(length);
        scale = alloc_d(length);

         /* Modified to deal with complex frequency vector */
        if (iscomplex(pl->pl_scale))
            for (i = 0; i < length; i++)
                scale[i] = realpart(pl->pl_scale->v_compdata[i]);
        else
            for (i = 0; i < length; i++)
                scale[i] = pl->pl_scale->v_realdata[i];

        /* use trapezoidal rule */
        outdata[0] = 0;
        for (i = 1; i < length; i++) {
            delta = scale[i] - scale[i - 1];
            outdata[i] = outdata[i - 1] + (indata[i] + indata[i - 1]) * delta / 2.;
        }

        tfree(scale);
        return (char*)outdata;
    }
}

void *
cx_group_delay(void *data, short int type, int length, int *newlength, short int *newtype, struct plot *pl, struct plot *newpl, int grouping)
{
    ngcomplex_t *cc = (ngcomplex_t *) data;
    double *v_phase = alloc_d(length);
    double *datos,adjust_final;
    double *group_delay = alloc_d(length);
    int i;
    /* char *datos_aux; */

    /* Check to see if we have the frequency vector for the derivative */
    if (!eq(pl->pl_scale->v_name, "frequency"))
    {
        fprintf(cp_err, "Internal error: cx_group_delay: need frequency based complex vector.\n");
        return (NULL);
    }

    if (type == VF_COMPLEX) {
        /*  accept continuous phase over 90° boundaries */
        double last_ph = cph(cc[0]);
        v_phase[0] = radtodeg(last_ph);
        for (i = 1; i < length; i++) {
            double ph = cph(cc[i]);
            last_ph = ph - (2 * M_PI) * floor((ph - last_ph) / (2 * M_PI) + 0.5);
            v_phase[i] = radtodeg(last_ph);
//            fprintf(stderr, "v_phase %e, cc %e %e\n", v_phase[i], cc[i].cx_real, cc[i].cx_imag);
        }
    }
    else
    {
        fprintf(cp_err, "Signal must be complex to calculate group delay\n");
        return (NULL);
    }

    type = VF_REAL;

    /* datos_aux = (char *)cx_deriv((char *)v_phase, type, length, newlength, newtype, pl, newpl, grouping);
     * datos = (double *) datos_aux;
     */
    datos = (double *)cx_deriv((char *)v_phase, type, length, newlength, newtype, pl, newpl, grouping);

    /* With this global variable I will change how to obtain the group delay because
     * it is defined as:
     *
     *  gd()=-dphase[rad]/dw[rad/s]
     *
     * if you have degrees in phase and frequency in Hz, must be taken into account
     *
     *  gd()=-dphase[deg]/df[Hz]/360
     *  gd()=-dphase[rad]/df[Hz]/(2*pi)
     */

    if(cx_degrees)
    {
        adjust_final=1.0/360;
    }
    else
    {
        adjust_final=1.0/(2*M_PI);
    }


    for (i = 0; i < length; i++)
    {
        group_delay[i] = -datos[i]*adjust_final;
    }

    /* Adjust to Real because the result is Real */
    *newtype = VF_REAL;


    /* Set the type of Vector to "Time" because the speed of group units' s'
     * The different types of vectors are INCLUDE \ Fte_cons.h
     */
    pl->pl_dvecs->v_type= SV_TIME;

    return ((char *) group_delay);
}


void *
cx_fft(void *data, short int type, int length, int *newlength, short int *newtype, struct plot *pl, struct plot *newpl, int grouping)
{
    int i, fpts, order;
    double span, scale, maxt;
    double *xscale;
    double *time = NULL, *win = NULL;
    ngcomplex_t *outdata = NULL;
    struct dvec  *sv;
    char   window[BSIZE_SP];
    double *realdata;

#ifdef HAVE_LIBFFTW3
    fftw_complex *inc;
    double *ind;
    fftw_complex *out = NULL;
    fftw_plan plan_forward = NULL;
#else
    int N, M;
    double *datax = NULL;
#endif

    if (grouping == 0)
        grouping = length;

    /* First do some sanity checks. */
    if (!pl || !pl->pl_scale || !newpl || !newpl->pl_scale) {
        fprintf(cp_err, "Internal error cx_fft: bad scale\n");
        return (NULL);
    }
    if ((type != VF_REAL) && (type != VF_COMPLEX)) {
        fprintf(cp_err, "Internal error cx_fft: argument has wrong data\n");
        return (NULL);
    }

#ifdef HAVE_LIBFFTW3
    if (type == VF_COMPLEX)
        fpts = length;
    else
        fpts = length/2 + 1;
#else
    /* size of fft input vector is power of two and larger or equal than spice vector */
    N = 1;
    M = 0;
    while (N < length) {
        N <<= 1;
        M++;
    }
    if (type == VF_COMPLEX)
        fpts = N;
    else
        fpts = N/2 + 1;
#endif

    *newtype = VF_COMPLEX;

    time = alloc_d(length);

    xscale = TMALLOC(double, length);

    if (pl->pl_scale->v_type == SV_TIME) { /* calculate the frequency from time */

        span = pl->pl_scale->v_realdata[length-1] - pl->pl_scale->v_realdata[0];

        for (i = 0; i<length; i++)
#ifdef HAVE_LIBFFTW3
            xscale[i] = i*1.0/span;
#else
            xscale[i] = i*1.0/span*length/N;
#endif
        for (i = 0; i<pl->pl_scale->v_length; i++)
            time[i] = pl->pl_scale->v_realdata[i];

    } else if (pl->pl_scale->v_type == SV_FREQUENCY) { /* take frequency from ac data and calculate time */

        /* Deal with complex frequency vector */
        if (iscomplex(pl->pl_scale)) {
            span = realpart(pl->pl_scale->v_compdata[pl->pl_scale->v_length-1]) - realpart(pl->pl_scale->v_compdata[0]);
            for (i = 0; i<pl->pl_scale->v_length; i++)
                xscale[i] = realpart(pl->pl_scale->v_compdata[i]);
        } else {
            span = pl->pl_scale->v_realdata[pl->pl_scale->v_length-1] - pl->pl_scale->v_realdata[0];
            for (i = 0; i<pl->pl_scale->v_length; i++)
                xscale[i] = pl->pl_scale->v_realdata[i];
        }

        for (i = 0; i < length; i++)
#ifdef HAVE_LIBFFTW3
            time[i] = i*1.0/span;
#else
            time[i] = i*1.0/span*length/N;
#endif

        span = time[length-1] - time[0];

    } else { /* there is no usable plot vector - using simple bins */

        for (i = 0; i < fpts; i++)
            xscale[i] = i;

        for (i = 0; i < length; i++)
            time[i] = i;

        span = time[length-1] - time[0];

    }

    win = TMALLOC(double, length);
    maxt = time[length-1];
    if (!cp_getvar("specwindow", CP_STRING, window, sizeof(window)))
        strcpy(window, "none");
    if (!cp_getvar("specwindoworder", CP_NUM, &order, 0))
        order = 2;
    if (order < 2)
        order = 2;

    if (fft_windows(window, win, time, length, maxt, span, order) == 0)
        goto done;

    /* create a new scale vector */
    sv = dvec_alloc(copy("fft_scale"),
                    SV_FREQUENCY,
                    VF_REAL | VF_PERMANENT | VF_PRINT,
                    fpts, xscale);
    vec_new(sv);

    if (type == VF_COMPLEX) { /* input vector is complex */

        ngcomplex_t *indata = (ngcomplex_t *) data;

#ifdef HAVE_LIBFFTW3

        printf("FFT: Time span: %g s, input length: %d\n", span, length);
        printf("FFT: Frequency resolution: %g Hz, output length: %d\n", 1.0/span, fpts);

        inc = fftw_malloc(sizeof(fftw_complex) * (unsigned int) length);
        out = fftw_malloc(sizeof(fftw_complex) * (unsigned int) fpts);

        for (i = 0; i < length; i++) {
            inc[i][0] = indata[i].cx_real * win[i];
            inc[i][1] = indata[i].cx_imag * win[i];
        }

        plan_forward = fftw_plan_dft_1d(fpts, inc, out, FFTW_FORWARD, FFTW_ESTIMATE);

        fftw_execute(plan_forward);

        *newlength = fpts;
        outdata = alloc_c(fpts);

        scale = (double) fpts;
        for (i = 0; i < fpts; i++) {
            outdata[i].cx_real = out[i][0]/scale;
            outdata[i].cx_imag = out[i][1]/scale;
        }

        fftw_free(inc);

#else /* Green's FFT */

        printf("FFT: Time span: %g s, input length: %d, zero padding: %d\n", span, length, N-length);
        printf("FFT: Frequency resolution: %g Hz, output length: %d\n", 1.0/span, N);

        datax = TMALLOC(double, 2*N);

        for (i = 0; i < length; i++) {
            datax[2*i] = indata[i].cx_real * win[i];
            datax[2*i+1] = indata[i].cx_imag * win[i];
        }
        for (i = length; i < N; i++) {
            datax[2*i] = 0.0;
            datax[2*i+1] = 0.0;
        }

        fftInit(M);
        ffts(datax, M, 1);
        fftFree();

        *newlength = N;
        outdata = alloc_c(N);

        scale = (double) N;
        for (i = 0; i < N; i++) {
            outdata[i].cx_real = datax[2*i]/scale;
            outdata[i].cx_imag = datax[2*i+1]/scale;
        }

#endif

    } else { /* input vector is real */

        realdata = (double *) data;
        *newlength = fpts;
        outdata = alloc_c(fpts);

#ifdef HAVE_LIBFFTW3

        printf("FFT: Time span: %g s, input length: %d\n", span, length);
        printf("FFT: Frequency resolution: %g Hz, output length: %d\n", 1.0/span, fpts);

        ind = fftw_malloc(sizeof(double) * (unsigned int) length);
        out = fftw_malloc(sizeof(fftw_complex) * (unsigned int) fpts);

        for (i = 0; i < length; i++)
            ind[i] = realdata[i] * win[i];

        plan_forward = fftw_plan_dft_r2c_1d(length, ind, out, FFTW_ESTIMATE);

        fftw_execute(plan_forward);

        scale = (double) fpts - 1.0;
        outdata[0].cx_real = out[0][0]/scale/2.0;
        outdata[0].cx_imag = 0.0;
        for (i = 1; i < fpts; i++) {
            outdata[i].cx_real = out[i][0]/scale;
            outdata[i].cx_imag = out[i][1]/scale;
        }

        fftw_free(ind);

#else /* Green's FFT */

        printf("FFT: Time span: %g s, input length: %d, zero padding: %d\n", span, length, N-length);
        printf("FFT: Frequency resolution: %g Hz, output length: %d\n", 1.0/span, fpts);

        datax = TMALLOC(double, N);

        for (i = 0; i < length; i++)
            datax[i] = realdata[i] * win[i];
        for (i = length; i < N; i++)
            datax[i] = 0.0;

        fftInit(M);
        rffts(datax, M, 1);
        fftFree();

        scale = (double) fpts - 1.0;
        /* Re(x[0]), Re(x[N/2]), Re(x[1]), Im(x[1]), Re(x[2]), Im(x[2]), ... Re(x[N/2-1]), Im(x[N/2-1]). */
        outdata[0].cx_real = datax[0]/scale/2.0;
        outdata[0].cx_imag = 0.0;
        for (i = 1; i < fpts-1; i++) {
            outdata[i].cx_real = datax[2*i]/scale;
            outdata[i].cx_imag = datax[2*i+1]/scale;
        }
        outdata[fpts-1].cx_real = datax[1]/scale;
        outdata[fpts-1].cx_imag = 0.0;

#endif

    }

done:
#ifdef HAVE_LIBFFTW3
    fftw_free(out);
    fftw_destroy_plan(plan_forward);
#else
    tfree(datax);
#endif
    tfree(time);
    tfree(win);

    return ((void *) outdata);
}


void *
cx_ifft(void *data, short int type, int length, int *newlength, short int *newtype, struct plot *pl, struct plot *newpl, int grouping)
{
    ngcomplex_t *indata = (ngcomplex_t *) data;
    int i, tpts;
    double span;
    double *xscale;
    ngcomplex_t *outdata = NULL;
    struct dvec  *sv;

#ifdef HAVE_LIBFFTW3
    fftw_complex *in;
    fftw_complex *out = NULL;
    fftw_plan plan_backward = NULL;
#else
    int N, M;
    double *datax = NULL;
    double scale;
#endif

    if (grouping == 0)
        grouping = length;

    /* First do some sanity checks. */
    if (!pl || !pl->pl_scale || !newpl || !newpl->pl_scale) {
        fprintf(cp_err, "Internal error cx_ifft: bad scale\n");
        return (NULL);
    }
    if ((type != VF_REAL) && (type != VF_COMPLEX)) {
        fprintf(cp_err, "Internal error cx_ifft: argument has wrong data\n");
        return (NULL);
    }

#ifdef HAVE_LIBFFTW3
    tpts = length;
#else
    /* size of ifft input vector is power of two and larger or equal than spice vector */
    N = 1;
    M = 0;
    while (N < length) {
        N <<= 1;
        M++;
    }
    tpts = N;
#endif

    if (pl->pl_scale->v_type == SV_TIME) { /* take the time from transient */

        /* output vector has same length as the plot scale vector */
        tpts = pl->pl_scale->v_length;

        xscale = TMALLOC(double, tpts);

        for (i = 0; i<tpts; i++)
            xscale[i] = pl->pl_scale->v_realdata[i];

    } else if (pl->pl_scale->v_type == SV_FREQUENCY) { /* calculate time from frequency */

        /* output vector has same length as the plot scale vector */
        tpts = pl->pl_scale->v_length;

        xscale = TMALLOC(double, tpts);

        /* Deal with complex frequency vector */
        if (iscomplex(pl->pl_scale))
            span = realpart(pl->pl_scale->v_compdata[tpts-1]) - realpart(pl->pl_scale->v_compdata[0]);
        else
            span = pl->pl_scale->v_realdata[tpts-1] - pl->pl_scale->v_realdata[0];

        for (i = 0; i<tpts; i++)
#ifdef HAVE_LIBFFTW3
            xscale[i] = i*1.0/span;
#else
            xscale[i] = i*1.0/span*length/N;
#endif

    } else {

        /* output vector has same length as input vector */
        tpts = length;

        xscale = TMALLOC(double, tpts);

        for (i = 0; i < tpts; i++)
            xscale[i] = i;

    }

    span = xscale[tpts-1] - xscale[0];

    /* create a new scale vector */
    sv = dvec_alloc(copy("ifft_scale"),
                    SV_TIME,
                    VF_REAL | VF_PERMANENT | VF_PRINT,
                    tpts, xscale);
    vec_new(sv);

    *newtype = VF_COMPLEX;
    *newlength = tpts;
    outdata = alloc_c(tpts);

#ifdef HAVE_LIBFFTW3

    printf("IFFT: Frequency span: %g Hz, input length: %d\n", 1/span*length, length);
    printf("IFFT: Time resolution: %g s, output length: %d\n", span/(tpts-1), tpts);

    in = fftw_malloc(sizeof(fftw_complex) * (unsigned int) tpts);
    out = fftw_malloc(sizeof(fftw_complex) * (unsigned int) tpts);

    for (i = 0; i < length; i++) {
        in[i][0] = indata[i].cx_real;
        in[i][1] = indata[i].cx_imag;
    }
    for (i = length; i < tpts; i++) {
        in[i][0] = 0.0;
        in[i][1] = 0.0;
    }

    plan_backward = fftw_plan_dft_1d(tpts, in, out, FFTW_BACKWARD, FFTW_ESTIMATE);

    fftw_execute(plan_backward);

    for (i = 0; i < tpts; i++) {
        outdata[i].cx_real = out[i][0];
        outdata[i].cx_imag = out[i][1];
    }

    fftw_free(in);
    fftw_destroy_plan(plan_backward);
    fftw_free(out);

#else /* Green's IFFT */

    printf("IFFT: Frequency span: %g Hz, input length: %d, zero padding: %d\n", 1/span*length, length, N-length);
    printf("IFFT: Time resolution: %g s, output length: %d\n", span/(tpts-1), tpts);

    datax = TMALLOC(double, 2*N);

    for (i = 0; i < length; i++) {
        datax[2*i] = indata[i].cx_real;
        datax[2*i+1] = indata[i].cx_imag;
    }
    for (i = length; i < N; i++) {
        datax[2*i] = 0.0;
        datax[2*i+1] = 0.0;
    }

    fftInit(M);
    iffts(datax, M, 1);
    fftFree();

    scale = (double) tpts;
    for (i = 0; i < tpts; i++) {
        outdata[i].cx_real = datax[2*i] * scale;
        outdata[i].cx_imag = datax[2*i+1] * scale;
    }

    tfree(datax);

#endif

    return ((void *) outdata);
}
