/**********
Copyright 2008 Holger Vogt.  All rights reserved.
Author:   2008 Holger Vogt
**********/

/*
 * Code to do fast fourier transform on data.
 */

#include "ngspice/ngspice.h"
#include "ngspice/ftedefs.h"
#include "ngspice/dvec.h"
#include "ngspice/sim.h"

#include "com_fft.h"
#include "variable.h"
#include "parse.h"
#include "../misc/misc_time.h"
#include "ngspice/fftext.h"

#ifdef HAVE_LIBFFTW3
#include "fftw3.h"
#endif


void
com_fft(wordlist *wl)
{
    ngcomplex_t **fdvec = NULL;
    double  **tdvec = NULL;
    double  *freq, *win = NULL, *time;
    double  span;
    int     fpts, i, j, length, ngood;
    struct dvec  *f, *vlist, *lv = NULL, *vec;
    struct pnode *pn, *names = NULL;
    char   window[BSIZE_SP];
    double maxt;
    double *in = NULL;

#ifdef HAVE_LIBFFTW3
    fftw_complex *out = NULL;
    fftw_plan plan_forward = NULL;
#else
    int N, M;
#endif

    int order;
    double scale;

    if (!plot_cur || !plot_cur->pl_scale) {
        fprintf(cp_err, "Error: no vectors loaded.\n");
        goto done;
    }
    if (!isreal(plot_cur->pl_scale) ||
        ((plot_cur->pl_scale)->v_type != SV_TIME)) {
        fprintf(cp_err, "Error: fft needs real time scale\n");
        goto done;
    }

    length = (plot_cur->pl_scale)->v_length;
    time = (plot_cur->pl_scale)->v_realdata;
    span = time[length-1] - time[0];

#ifdef HAVE_LIBFFTW3
    fpts = length/2 + 1;
#else
    /* size of fft input vector is power of two and larger or equal than spice vector */
    N = 1;
    M = 0;
    while (N < length) {
        N <<= 1;
        M++;
    }
    fpts = N/2 + 1;
#endif

    win = TMALLOC(double, length);
    maxt = time[length-1];
    if (!cp_getvar("specwindow", CP_STRING, window, sizeof(window)))
        strcpy(window, "hanning");
    if (!cp_getvar("specwindoworder", CP_NUM, &order, 0))
        order = 2;
    if (order < 2)
        order = 2;

    if (fft_windows(window, win, time, length, maxt, span, order) == 0)
        goto done;

    names = ft_getpnames_quotes(wl, TRUE);
    vlist = NULL;
    ngood = 0;
    for (pn = names; pn; pn = pn->pn_next) {
        vec = ft_evaluate(pn);
        for (; vec; vec = vec->v_link2) {

            if (vec->v_length != length) {
                fprintf(cp_err, "Error: lengths of %s vectors don't match: %d, %d\n",
                        vec->v_name, vec->v_length, length);
                continue;
            }

            if (!isreal(vec)) {
                fprintf(cp_err, "Error: %s isn't real!\n", vec->v_name);
                continue;
            }

            if (vec->v_type == SV_TIME) {
                continue;
            }

            if (!vlist)
                vlist = vec;
            else
                lv->v_link2 = vec;

            lv = vec;
            ngood++;
        }
    }

    if (!ngood)
        goto done;

    plot_cur = plot_alloc("spectrum");
    plot_cur->pl_next = plot_list;
    plot_list = plot_cur;
    plot_cur->pl_title = copy((plot_cur->pl_next)->pl_title);
    plot_cur->pl_name = copy("Spectrum");
    plot_cur->pl_date = copy(datestring());

    f = dvec_alloc(copy("frequency"),
                   SV_FREQUENCY,
                   VF_REAL | VF_PERMANENT | VF_PRINT,
                   fpts, NULL);
    vec_new(f);
    freq = f->v_realdata;

    for (i = 0; i<fpts; i++)
#ifdef HAVE_LIBFFTW3
        freq[i] = i*1.0/span;
#else
        freq[i] = i*1.0/span*length/N;
#endif

    tdvec = TMALLOC(double  *, ngood);
    fdvec = TMALLOC(ngcomplex_t *, ngood);
    for (i = 0, vec = vlist; i<ngood; i++) {
        tdvec[i] = vec->v_realdata; /* real input data */
        f = dvec_alloc(vec_basename(vec),
                       SV_NOTYPE,
                       VF_COMPLEX | VF_PERMANENT,
                       fpts, NULL);
        vec_new(f);
        fdvec[i] = f->v_compdata; /* complex output data */
        vec = vec->v_link2;
    }

#ifdef HAVE_LIBFFTW3

    printf("FFT: Time span: %g s, input length: %d\n", span, length);
    printf("FFT: Frequency resolution: %g Hz, output length: %d\n", 1.0/span, fpts);

    in = fftw_malloc(sizeof(double) * (unsigned int) length);
    out = fftw_malloc(sizeof(fftw_complex) * (unsigned int) fpts);

    for (j = 0; j < length; j++)
        in[j] = tdvec[0][j]*win[j];

    /* data have same type and length - so we need only one plan */
    plan_forward = fftw_plan_dft_r2c_1d(length, in, out, FFTW_ESTIMATE); 

    for (i = 0; i<ngood; i++) {

        if (i > 0) {
            for (j = 0; j < length; j++)
                in[j] = tdvec[i][j]*win[j];
        }

        fftw_execute(plan_forward);

        scale = (double) fpts - 1.0;
        fdvec[i][0].cx_real = out[0][0]/scale/2.0;
        fdvec[i][0].cx_imag = 0.0;
        for (j = 1; j < fpts; j++) {
            fdvec[i][j].cx_real = out[j][0]/scale;
            fdvec[i][j].cx_imag = out[j][1]/scale;
        }

    }

    fftw_destroy_plan(plan_forward);

    fftw_free(in);
    fftw_free(out);

#else /* Green's FFT */

    printf("FFT: Time span: %g s, input length: %d, zero padding: %d\n", span, length, N-length);
    printf("FFT: Frequency resolution: %g Hz, output length: %d\n", 1.0/span, fpts);

    for (i = 0; i<ngood; i++) {

        in = TMALLOC(double, N);
        for (j = 0; j < length; j++) {
            in[j] = tdvec[i][j]*win[j];
        }
        for (j = length; j < N; j++) {
            in[j] = 0.0;
        }

        fftInit(M);
        rffts(in, M, 1);
        fftFree();

        scale = (double) fpts - 1.0;
        /* Re(x[0]), Re(x[N/2]), Re(x[1]), Im(x[1]), Re(x[2]), Im(x[2]), ... Re(x[N/2-1]), Im(x[N/2-1]). */
        fdvec[i][0].cx_real = in[0]/scale/2.0;
        fdvec[i][0].cx_imag = 0.0;
        for (j = 1; j < fpts-1; j++) {
            fdvec[i][j].cx_real = in[2*j]/scale;
            fdvec[i][j].cx_imag = in[2*j+1]/scale;
        }
        fdvec[i][fpts-1].cx_real = in[1]/scale;
        fdvec[i][fpts-1].cx_imag = 0.0;

        tfree(in);

    }

#endif

done:
    tfree(tdvec);
    tfree(fdvec);
    tfree(win);

    free_pnode(names);
}


void
com_psd(wordlist *wl)
{
    ngcomplex_t **fdvec = NULL;
    double  **tdvec = NULL;
    double  *freq, *win = NULL, *time;
    double  span, noipower;
    int ngood, fpts, i, j, jj, length, smooth, hsmooth;
    char    *s;
    struct dvec  *f, *vlist, *lv = NULL, *vec;
    struct pnode *pn, *names = NULL;
    char   window[BSIZE_SP];
    double maxt, intres;

#ifdef HAVE_LIBFFTW3
    double *in = NULL;
    fftw_complex *out = NULL;
    fftw_plan plan_forward = NULL;
#else
    int N, M;
#endif

    double *reald = NULL;
    double sum;
    int order;

    if (!plot_cur || !plot_cur->pl_scale) {
        fprintf(cp_err, "Error: no vectors loaded.\n");
        goto done;
    }
    if (!isreal(plot_cur->pl_scale) ||
        ((plot_cur->pl_scale)->v_type != SV_TIME)) {
        fprintf(cp_err, "Error: fft needs real time scale\n");
        goto done;
    }

    length = (plot_cur->pl_scale)->v_length;
    time = (plot_cur->pl_scale)->v_realdata;
    span = time[length-1] - time[0];

    // get filter length from parameter input
    s = wl->wl_word;
    {
        double val;
        if (ft_numparse(&s, FALSE, &val) <= 0 || val < 1.0) {
            fprintf(cp_out, "Number of averaged data points:  1\n");
            smooth = 1;
        }
        else {
            smooth = (int) val;
        }
    }

    wl = wl->wl_next;

#ifdef HAVE_LIBFFTW3
    fpts = length/2 + 1;
#else
    /* size of fft input vector is power of two and larger or equal than spice vector */
    N = 1;
    M = 0;
    while (N < length) {
        N <<= 1;
        M++;
    }
    fpts = N/2 + 1;
#endif

    win = TMALLOC(double, length);
    maxt = time[length-1];
    if (!cp_getvar("specwindow", CP_STRING, window, sizeof(window)))
        strcpy(window, "hanning");
    if (!cp_getvar("specwindoworder", CP_NUM, &order, 0))
        order = 2;
    if (order < 2)
        order = 2;

    if (fft_windows(window, win, time, length, maxt, span, order) == 0)
        goto done;

    names = ft_getpnames_quotes(wl, TRUE);
    vlist = NULL;
    ngood = 0;
    for (pn = names; pn; pn = pn->pn_next) {
        vec = ft_evaluate(pn);
        for (; vec; vec = vec->v_link2) {

            if (vec->v_length != (int)length) {
                fprintf(cp_err, "Error: lengths of %s vectors don't match: %d, %d\n",
                        vec->v_name, vec->v_length, length);
                continue;
            }

            if (!isreal(vec)) {
                fprintf(cp_err, "Error: %s isn't real!\n", vec->v_name);
                continue;
            }

            if (vec->v_type == SV_TIME) {
                continue;
            }

            if (!vlist)
                vlist = vec;
            else
                lv->v_link2 = vec;

            lv = vec;
            ngood++;
        }
    }

    if (!ngood)
        goto done;

    plot_cur = plot_alloc("spectrum");
    plot_cur->pl_next = plot_list;
    plot_list = plot_cur;
    plot_cur->pl_title = copy((plot_cur->pl_next)->pl_title);
    plot_cur->pl_name = copy("PSD");
    plot_cur->pl_date = copy(datestring());

    f = dvec_alloc(copy("frequency"),
                   SV_FREQUENCY,
                   VF_REAL | VF_PERMANENT | VF_PRINT,
                   fpts, NULL);
    vec_new(f);
    freq = f->v_realdata;

    for (i = 0; i < fpts; i++)
#ifdef HAVE_LIBFFTW3
        freq[i] = i*1./span;
#else
        freq[i] = i*1./span*length/N;
#endif

    tdvec = TMALLOC(double*, ngood);
    fdvec = TMALLOC(ngcomplex_t*, ngood);
    for (i = 0, vec = vlist; i<ngood; i++) {
        tdvec[i] = vec->v_realdata; /* real input data */
        f = dvec_alloc(vec_basename(vec),
                       SV_NOTYPE, //vec->v_type
                       VF_COMPLEX | VF_PERMANENT,
                       fpts, NULL);
        vec_new(f);
        fdvec[i] = f->v_compdata; /* complex output data */
        vec = vec->v_link2;
    }

#ifdef HAVE_LIBFFTW3

    printf("PSD: Time span: %g s, input length: %d\n", span, length);
    printf("PSD: Frequency resolution: %g Hz, output length: %d\n", 1.0/span, fpts);

    reald = TMALLOC(double, fpts);

    in = fftw_malloc(sizeof(double) * (unsigned int) length);
    out = fftw_malloc(sizeof(fftw_complex) * (unsigned int) fpts);

    for (j = 0; j < length; j++)
        in[j] = tdvec[0][j]*win[j];

    /* data have same type and length - so we need only one plan */
    plan_forward = fftw_plan_dft_r2c_1d(length, in, out, FFTW_ESTIMATE);

    for (i = 0; i<ngood; i++) {

        if (i > 0) {
            for (j = 0; j < length; j++)
                in[j] = tdvec[i][j]*win[j];
        }

        fftw_execute(plan_forward);

        intres = (double)length * (double)length;
        fdvec[i][0].cx_real = out[0][0]*out[0][0]/intres;
        fdvec[i][0].cx_imag = 0;
        noipower = fdvec[i][0].cx_real;
        for (j = 1; j < fpts-1; j++) {
            fdvec[i][j].cx_real = 2.* (out[j][0]*out[j][0] + out[j][1]*out[j][1])/intres;
            fdvec[i][j].cx_imag = 0;
            noipower += fdvec[i][j].cx_real;
            if (!finite(noipower))
                break;
        }
        fdvec[i][fpts-1].cx_real = out[fpts-1][0]*out[fpts-1][0]/intres;
        fdvec[i][fpts-1].cx_imag = 0;
        noipower += fdvec[i][fpts-1].cx_real;

#else /* Green's FFT */

    printf("PSD: Time span: %g s, input length: %d, zero padding: %d\n", span, length, N-length);
    printf("PSD: Frequency resolution: %g Hz, output length: %d\n", 1.0/span, fpts);

    reald = TMALLOC(double, N);

    for (i = 0; i<ngood; i++) {

        for (j = 0; j < length; j++) {
            reald[j] = (tdvec[i][j]*win[j]);
        }
        for (j = length; j < N; j++) {
            reald[j] = 0.;
        }

        fftInit(M);
        rffts(reald, M, 1);
        fftFree();

        /* Re(x[0]), Re(x[N/2]), Re(x[1]), Im(x[1]), Re(x[2]), Im(x[2]), ... Re(x[N/2-1]), Im(x[N/2-1]). */
        intres = (double)N * (double)N;
        fdvec[i][0].cx_real = reald[0]*reald[0]/intres;
        fdvec[i][0].cx_imag = 0;
        noipower = fdvec[i][0].cx_real;
        for (j = 1; j < fpts-1; j++) {
            jj = j<<1;
            fdvec[i][j].cx_real = 2.* (reald[jj]*reald[jj] + reald[jj + 1]*reald[jj + 1])/intres;
            fdvec[i][j].cx_imag = 0;
            noipower += fdvec[i][j].cx_real;
            if (!finite(noipower))
                break;
        }
        fdvec[i][fpts-1].cx_real = reald[1]*reald[1]/intres;
        fdvec[i][fpts-1].cx_imag = 0;
        noipower += fdvec[i][fpts-1].cx_real;

#endif

        printf("Total noise power up to Nyquist frequency %5.3e Hz: %e V^2 (or A^2), \nNoise voltage or current: %e V (or A)\n",
               freq[fpts-1], noipower, sqrt(noipower));

        /* smoothing with rectangular window of width "smooth",
           plotting V^2/Hz or I^2/Hz */
        hsmooth = smooth>>1;
        for (j = 0; j < hsmooth; j++) {
            sum = 0.;
            for (jj = 0; jj < hsmooth + j; jj++)
                sum += fdvec[i][jj].cx_real;
            sum /= (double) (hsmooth + j);
            reald[j] = sum;
        }
        for (j = hsmooth; j < fpts-hsmooth; j++) {
            sum = 0.;
            for (jj = 0; jj < smooth; jj++)
                sum += fdvec[i][j-hsmooth+jj].cx_real;
            sum /= (double) smooth;
            reald[j] = sum;
        }
        for (j = fpts-hsmooth; j < fpts; j++) {
            sum = 0.;
            for (jj = 0; jj < fpts+hsmooth-j-1; jj++)
                sum += fdvec[i][j-hsmooth+jj+1].cx_real;
            sum /= (double) (fpts+hsmooth-j-1);
            reald[j] = sum;
        }
        for (j = 0; j < fpts; j++)
            fdvec[i][j].cx_real = reald[j] * (double)fpts / freq[fpts - 1];
    }

#ifdef HAVE_LIBFFTW3
    fftw_destroy_plan(plan_forward);
    fftw_free(in);
    fftw_free(out);
#endif
done:
    tfree(tdvec);
    tfree(fdvec);
    tfree(win);

    tfree(reald);

    free_pnode(names);
}
