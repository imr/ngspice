/**********
Copyright 2008 Holger Vogt.  All rights reserved.
Author:   2008 Holger Vogt
**********/

/*
 * Code to do fast fourier transform on data.
 */

#define GREEN /* select fast Green's fft */

#include "ngspice/ngspice.h"
#include "ngspice/ftedefs.h"
#include "ngspice/dvec.h"
#include "ngspice/sim.h"

#include "com_fft.h"
#include "variable.h"
#include "parse.h"
#include "../misc/misc_time.h"
#include "ngspice/fftext.h"


#ifndef GREEN
static void fftext(double*, double*, long int, long int, int);
#endif


void
com_fft(wordlist *wl)
{
    ngcomplex_t **fdvec = NULL;
    double  **tdvec = NULL;
    double  *freq, *win = NULL, *time;
    double  span;
    int     fpts, i, j, tlen, ngood;
    struct dvec  *f, *vlist, *lv = NULL, *vec;
    struct pnode *pn, *names = NULL;
    char   window[BSIZE_SP];
    double maxt;

#ifdef GREEN
    int mm;
#endif

    double *reald = NULL, *imagd = NULL;
    int size, sign, order;
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

    tlen = (plot_cur->pl_scale)->v_length;
    time = (plot_cur->pl_scale)->v_realdata;
    span = time[tlen-1] - time[0];

#ifdef GREEN
    // size of input vector is power of two and larger than spice vector
    size = 1;
    mm = 0;
    while (size < tlen) {
        size <<= 1;
        mm++;
    }
#else
    /* size of input vector is power of two and larger than spice vector */
    size = 1;
    while (size < tlen)
        size *= 2;
#endif
    /* output vector has length of size/2 */
    fpts = size/2;

    win = TMALLOC(double, tlen);
    maxt = time[tlen-1];
    if (!cp_getvar("specwindow", CP_STRING, window))
        strcpy(window, "blackman");
    if (!cp_getvar("specwindoworder", CP_NUM, &order))
        order = 2;
    if (order < 2)
        order = 2;

    if (fft_windows(window, win, time, tlen, maxt, span, order) == 0)
        goto done;

    names = ft_getpnames(wl, TRUE);
    vlist = NULL;
    ngood = 0;
    for (pn = names; pn; pn = pn->pn_next) {
        vec = ft_evaluate(pn);
        for (; vec; vec = vec->v_link2) {

            if (vec->v_length != tlen) {
                fprintf(cp_err, "Error: lengths of %s vectors don't match: %d, %d\n",
                        vec->v_name, vec->v_length, tlen);
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

    freq = TMALLOC(double, fpts);
    f = alloc(struct dvec);
    ZERO(f, struct dvec);
    f->v_name = copy("frequency");
    f->v_type = SV_FREQUENCY;
    f->v_flags = (VF_REAL | VF_PERMANENT | VF_PRINT);
    f->v_length = fpts;
    f->v_realdata = freq;
    vec_new(f);

    for (i = 0; i<fpts; i++)
        freq[i] = i*1.0/span*tlen/size;

    tdvec = TMALLOC(double  *, ngood);
    fdvec = TMALLOC(ngcomplex_t *, ngood);
    for (i = 0, vec = vlist; i<ngood; i++) {
        tdvec[i] = vec->v_realdata; /* real input data */
        fdvec[i] = TMALLOC(ngcomplex_t, fpts); /* complex output data */
        f = alloc(struct dvec);
        ZERO(f, struct dvec);
        f->v_name = vec_basename(vec);
        f->v_type = SV_NOTYPE;
        f->v_flags = (VF_COMPLEX | VF_PERMANENT);
        f->v_length = fpts;
        f->v_compdata = fdvec[i];
        vec_new(f);
        vec = vec->v_link2;
    }

    sign = 1;

    printf("FFT: Time span: %g s, input length: %d, zero padding: %d\n", span, size, size-tlen);
    printf("FFT: Freq. resolution: %g Hz, output length: %d\n", 1.0/span*tlen/size, fpts);

    reald = TMALLOC(double, size);
    imagd = TMALLOC(double, size);
    for (i = 0; i<ngood; i++) {
        for (j = 0; j < tlen; j++) {
            reald[j] = tdvec[i][j]*win[j];
            imagd[j] = 0.0;
        }
        for (j = tlen; j < size; j++) {
            reald[j] = 0.0;
            imagd[j] = 0.0;
        }
#ifdef GREEN
        // Green's FFT
        fftInit(mm);
        rffts(reald, mm, 1);
        fftFree();
        scale = size;
        /* Re(x[0]), Re(x[N/2]), Re(x[1]), Im(x[1]), Re(x[2]), Im(x[2]), ... Re(x[N/2-1]), Im(x[N/2-1]). */
        for (j = 0; j < fpts; j++) {
            fdvec[i][j].cx_real = reald[2*j]/scale;
            fdvec[i][j].cx_imag = reald[2*j+1]/scale;
        }
        fdvec[i][0].cx_imag = 0;
#else
        fftext(reald, imagd, size, tlen, sign);
        scale = 0.66;

        for (j = 0; j < fpts; j++) {
            fdvec[i][j].cx_real = reald[j]/scale;
            fdvec[i][j].cx_imag = imagd[j]/scale;
        }
#endif
    }

done:
    tfree(reald);
    tfree(imagd);

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
    double  *freq, *win = NULL, *time, *ave;
    double  span, noipower;
    int     mm;
    int size, ngood, fpts, i, j, tlen, jj, smooth, hsmooth;
    char    *s;
    struct dvec  *f, *vlist, *lv = NULL, *vec;
    struct pnode *pn, *names = NULL;
    char   window[BSIZE_SP];
    double maxt;

    double *reald = NULL, *imagd = NULL;
    int sign;
    double scaling, sum;
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

    tlen = (plot_cur->pl_scale)->v_length;
    time = (plot_cur->pl_scale)->v_realdata;
    span = time[tlen-1] - time[0];

    // get filter length from parameter input
    s = wl->wl_word;
    ave = ft_numparse(&s, FALSE);
    if (!ave || (*ave < 1.0)) {
        fprintf(cp_out, "Number of averaged data points:  %d\n", 1);
        smooth = 1;
    } else {
        smooth = (int)(*ave);
    }

    wl = wl->wl_next;

    // size of input vector is power of two and larger than spice vector
    size = 1;
    mm = 0;
    while (size < tlen) {
        size <<= 1;
        mm++;
    }

    // output vector has length of size/2
    fpts = size>>1;

    win = TMALLOC(double, tlen);
    maxt = time[tlen-1];
    if (!cp_getvar("specwindow", CP_STRING, window))
        strcpy(window, "blackman");
    if (!cp_getvar("specwindoworder", CP_NUM, &order))
        order = 2;
    if (order < 2)
        order = 2;

    if (fft_windows(window, win, time, tlen, maxt, span, order) == 0)
        goto done;

    names = ft_getpnames(wl, TRUE);
    vlist = NULL;
    ngood = 0;
    for (pn = names; pn; pn = pn->pn_next) {
        vec = ft_evaluate(pn);
        for (; vec; vec = vec->v_link2) {

            if (vec->v_length != (int)tlen) {
                fprintf(cp_err, "Error: lengths of %s vectors don't match: %d, %d\n",
                        vec->v_name, vec->v_length, tlen);
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

    freq = TMALLOC(double, fpts + 1);
    f = alloc(struct dvec);
    ZERO(f, struct dvec);
    f->v_name = copy("frequency");
    f->v_type = SV_FREQUENCY;
    f->v_flags = (VF_REAL | VF_PERMANENT | VF_PRINT);
    f->v_length = fpts;
    f->v_realdata = freq;
    vec_new(f);

    for (i = 0; i <= fpts; i++)
        freq[i] = i*1./span*tlen/size;

    tdvec = TMALLOC(double*, ngood);
    fdvec = TMALLOC(ngcomplex_t*, ngood);
    for (i = 0, vec = vlist; i<ngood; i++) {
        tdvec[i] = vec->v_realdata; /* real input data */
        fdvec[i] = TMALLOC(ngcomplex_t, fpts + 1); /* complex output data */
        f = alloc(struct dvec);
        ZERO(f, struct dvec);
        f->v_name = vec_basename(vec);
        f->v_type = SV_NOTYPE; //vec->v_type;
        f->v_flags = (VF_COMPLEX | VF_PERMANENT);
        f->v_length = fpts + 1;
        f->v_compdata = fdvec[i];
        vec_new(f);
        vec = vec->v_link2;
    }

    printf("PSD: Time span: %g s, input length: %d, zero padding: %d\n", span, size, size-tlen);
    printf("PSD: Freq. resolution: %g Hz, output length: %d\n", 1.0/span*tlen/size, fpts);

    sign = 1;

    reald = TMALLOC(double, size);
    imagd = TMALLOC(double, size);

    // scale = 0.66;

    for (i = 0; i<ngood; i++) {
        double intres;
        for (j = 0; j < tlen; j++) {
            reald[j] = (tdvec[i][j]*win[j]);
            imagd[j] = 0.;
        }
        for (j = tlen; j < size; j++) {
            reald[j] = 0.;
            imagd[j] = 0.;
        }

        // Green's FFT
        fftInit(mm);
        rffts(reald, mm, 1);
        fftFree();
        scaling = size;

        /* Re(x[0]), Re(x[N/2]), Re(x[1]), Im(x[1]), Re(x[2]), Im(x[2]), ... Re(x[N/2-1]), Im(x[N/2-1]). */
        intres = (double)size * (double)size;
        noipower = fdvec[i][0].cx_real = reald[0]*reald[0]/intres;
        fdvec[i][fpts].cx_real = reald[1]*reald[1]/intres;
        noipower += fdvec[i][fpts-1].cx_real;
        for (j = 1; j < fpts; j++) {
            jj = j<<1;
            fdvec[i][j].cx_real = 2.* (reald[jj]*reald[jj] + reald[jj + 1]*reald[jj + 1])/intres;
            fdvec[i][j].cx_imag = 0;
            noipower += fdvec[i][j].cx_real;
            if (!finite(noipower))
                break;
        }

        printf("Total noise power up to Nyquist frequency %5.3e Hz:\n%e V^2 (or A^2), \nnoise voltage or current %e V (or A)\n",
               freq[fpts], noipower, sqrt(noipower));

        /* smoothing with rectangular window of width "smooth",
           plotting V/sqrt(Hz) or I/sqrt(Hz) */
        if (smooth < 1)
            continue;

        hsmooth = smooth>>1;
        for (j = 0; j < hsmooth; j++) {
            sum = 0.;
            for (jj = 0; jj < hsmooth + j; jj++)
                sum += fdvec[i][jj].cx_real;
            sum /= (hsmooth + j);
            reald[j] = (sqrt(sum)/scaling);
        }
        for (j = hsmooth; j < fpts-hsmooth; j++) {
            sum = 0.;
            for (jj = 0; jj < smooth; jj++)
                sum += fdvec[i][j-hsmooth+jj].cx_real;
            sum /= smooth;
            reald[j] = (sqrt(sum)/scaling);
        }
        for (j = fpts-hsmooth; j < fpts; j++) {
            sum = 0.;
            for (jj = 0; jj < smooth; jj++)
                sum += fdvec[i][j-hsmooth+jj].cx_real;
            sum /= (fpts - j + hsmooth - 1);
            reald[j] = (sqrt(sum)/scaling);
        }
        for (j = 0; j < fpts; j++)
            fdvec[i][j].cx_real = reald[j];
    }

done:
    free(reald);
    free(imagd);

    tfree(tdvec);
    tfree(fdvec);
    tfree(win);

    free_pnode(names);
}


#ifndef GREEN

static void
fftext(double *x, double *y, long int n, long int nn, int dir)
{
    /*
      http://local.wasp.uwa.edu.au/~pbourke/other/dft/
      download 22.05.08
      Used with permission from the author Paul Bourke
    */

    /*
      This computes an in-place complex-to-complex FFT
      x and y are the real and imaginary arrays
      n is the number of points, has to be to the power of 2
      nn is the number of points w/o zero padded values
      dir =  1 gives forward transform
      dir = -1 gives reverse transform
    */

    long i, i1, j, k, i2, l, l1, l2;
    double c1, c2, tx, ty, t1, t2, u1, u2, z;
    int m = 0, mm = 1;

    /* get the exponent to the base of 2 from the number of points */
    while (mm < n) {
        mm *= 2;
        m++;
    }

    /* Do the bit reversal */
    i2 = n >> 1;
    j = 0;
    for (i = 0; i < n-1; i++) {
        if (i < j) {
            tx = x[i];
            ty = y[i];
            x[i] = x[j];
            y[i] = y[j];
            x[j] = tx;
            y[j] = ty;
        }
        k = i2;
        while (k <= j) {
            j -= k;
            k >>= 1;
        }
        j += k;
    }

    /* Compute the FFT */
    c1 = -1.0;
    c2 = 0.0;
    l2 = 1;
    for (l = 0; l < m; l++) {
        l1 = l2;
        l2 <<= 1;
        u1 = 1.0;
        u2 = 0.0;
        for (j = 0; j < l1; j++) {
            for (i = j; i < n; i += l2) {
                i1 = i + l1;
                t1 = u1 * x[i1] - u2 * y[i1];
                t2 = u1 * y[i1] + u2 * x[i1];
                x[i1] = x[i] - t1;
                y[i1] = y[i] - t2;
                x[i] += t1;
                y[i] += t2;
            }
            z =  u1 * c1 - u2 * c2;
            u2 = u1 * c2 + u2 * c1;
            u1 = z;
        }
        c2 = sqrt((1.0 - c1) / 2.0);
        if (dir == 1)
            c2 = -c2;
        c1 = sqrt((1.0 + c1) / 2.0);
    }

    /* Scaling for forward transform */
    if (dir == 1) {
        double scale = 1.0 / nn;
        for (i = 0; i < n; i++) {
            x[i] *= scale; /* don't consider zero padded values */
            y[i] *= scale;
        }
    }
}

#endif /* GREEN */
