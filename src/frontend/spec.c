/**********
Copyright 1994 Macquarie University, Sydney Australia.  All rights reserved.
Author:   1994 Anthony E. Parker, Department of Electronics, Macquarie Uni.
**********/

/*
 * Code to do fourier transforms on data.
 */

#include "ngspice/ngspice.h"
#include "ngspice/ftedefs.h"
#include "ngspice/dvec.h"
#include "ngspice/sim.h"

#include "spec.h"
#include "parse.h"
#include "variable.h"
#include "ngspice/missing_math.h"
#include "../misc/misc_time.h"


void
com_spec(wordlist *wl)
{
    ngcomplex_t **fdvec;
    double  **tdvec;
    double  *freq, *win, *time, *dc;
    double  startf, stopf, stepf, span;
    int     fpts, i, j, k, tlen, ngood;
    bool    trace;
    char    *s;
    struct dvec  *f, *vlist, *lv = NULL, *vec;
    struct pnode *pn, *names;

    if (!plot_cur || !plot_cur->pl_scale) {
        fprintf(cp_err, "Error: no vectors loaded.\n");
        return;
    }

    if (!isreal(plot_cur->pl_scale) ||
        ((plot_cur->pl_scale)->v_type != SV_TIME)) {
        fprintf(cp_err, "Error: spec needs real time scale\n");
        return;
    }

    s = wl->wl_word;
    tlen = (plot_cur->pl_scale)->v_length;
    if ((freq = ft_numparse(&s, FALSE)) == NULL || (*freq < 0.0)) {
        fprintf(cp_err, "Error: bad start freq %s\n", wl->wl_word);
        return;
    }
    startf = *freq;

    wl = wl->wl_next;
    s = wl->wl_word;
    if ((freq = ft_numparse(&s, FALSE)) == NULL || (*freq <= startf)) {
        fprintf(cp_err, "Error: bad stop freq %s\n", wl->wl_word);
        return;
    }
    stopf = *freq;

    wl = wl->wl_next;
    s = wl->wl_word;
    if ((freq = ft_numparse(&s, FALSE)) == NULL || !(*freq <= (stopf-startf))) {
        fprintf(cp_err, "Error: bad step freq %s\n", wl->wl_word);
        return;
    }
    stepf = *freq;

    wl = wl->wl_next;
    time = (plot_cur->pl_scale)->v_realdata;
    span = time[tlen-1] - time[0];
    if (stopf > 0.5*tlen/span) {
        fprintf(cp_err,
                "Error: nyquist limit exceeded, try stop freq less than %e Hz\n",
                tlen/2/span);
        return;
    }
    span = ((int)(span*stepf*1.000000000001))/stepf;
    if (span > 0) {
        startf = (int)(startf/stepf*1.000000000001) * stepf;
        fpts = (int)((stopf - startf)/stepf + 1.);
        if (stopf > startf + (fpts-1)*stepf)
            fpts++;
    } else {
        fprintf(cp_err, "Error: time span limits step freq to %1.1e Hz\n",
                1/(time[tlen-1] - time[0]));
        return;
    }
    win = TMALLOC(double, tlen);
    {
        char   window[BSIZE_SP];
        double maxt = time[tlen-1];
        if (!cp_getvar("specwindow", CP_STRING, window))
            strcpy(window, "hanning");
        if (eq(window, "none"))
            for (i = 0; i < tlen; i++)
                win[i] = 1;
        else if (eq(window, "rectangular"))
            for (i = 0; i < tlen; i++) {
                if (maxt-time[i] > span) {
                    win[i] = 0;
                } else {
                    win[i] = 1;
                }
            }
        else if (eq(window, "hanning") || eq(window, "cosine"))
            for (i = 0; i < tlen; i++) {
                if (maxt-time[i] > span) {
                    win[i] = 0;
                } else {
                    win[i] = 1 - cos(2*M_PI*(time[i]-maxt)/span);
                }
            }
        else if (eq(window, "hamming"))
            for (i = 0; i < tlen; i++) {
                if (maxt-time[i] > span) {
                    win[i] = 0;
                } else {
                    win[i] = 1 - 0.92/1.08*cos(2*M_PI*(time[i]-maxt)/span);
                }
            }
        else if (eq(window, "triangle") || eq(window, "bartlet"))
            for (i = 0; i < tlen; i++) {
                if (maxt-time[i] > span) {
                    win[i] = 0;
                } else {
                    win[i] = 2 - fabs(2+4*(time[i]-maxt)/span);
                }
            }
        else if (eq(window, "blackman")) {
            int order;
            if (!cp_getvar("specwindoworder", CP_NUM, &order))
                order = 2;
            if (order < 2)      /* only order 2 supported here */
                order = 2;
            for (i = 0; i < tlen; i++) {
                if (maxt-time[i] > span) {
                    win[i] = 0;
                } else {
                    win[i]  = 1;
                    win[i] -= 0.50/0.42*cos(2*M_PI*(time[i]-maxt)/span);
                    win[i] += 0.08/0.42*cos(4*M_PI*(time[i]-maxt)/span);
                }
            }
        } else if (eq(window, "gaussian")) {
            int order;
            double scale;
            if (!cp_getvar("specwindoworder", CP_NUM, &order))
                order = 2;
            if (order < 2)
                order = 2;
            scale = pow(2*M_PI/order, 0.5)*(0.5-erfc(pow(order, 0.5)));
            for (i = 0; i < tlen; i++) {
                if (maxt-time[i] > span) {
                    win[i] = 0;
                } else {
                    win[i] = exp(-0.5*order*(1-2*(maxt-time[i])/span)
                                 *(1-2*(maxt-time[i])/span))/scale;
                }
            }
        } else {
            fprintf(cp_err, "Warning: unknown window type %s\n", window);
            tfree(win);
            return;
        }
    }

    names = ft_getpnames(wl, TRUE);
    vlist = NULL;
    ngood = 0;
    for (pn = names; pn; pn = pn->pn_next) {
        vec = ft_evaluate(pn);
        for (; vec; vec = vec->v_link2) {

            if (vec->v_length != tlen) {
                fprintf(cp_err, "Error: lengths don't match: %d, %d\n",
                        vec->v_length, tlen);
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
    free_pnode_o(names); /* h_vogt 081206 */
    if (!ngood) {
        tfree(win);
        return;
    }

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

    tdvec = TMALLOC(double  *, ngood);
    fdvec = TMALLOC(ngcomplex_t *, ngood);
    for (i = 0, vec = vlist; i < ngood; i++) {
        tdvec[i] = vec->v_realdata;
        fdvec[i] = TMALLOC(ngcomplex_t, fpts);
        f = alloc(struct dvec);
        ZERO(f, struct dvec);
        f->v_name = vec_basename(vec);
        f->v_type = vec->v_type;
        f->v_flags = (VF_COMPLEX | VF_PERMANENT);
        f->v_length = fpts;
        f->v_compdata = fdvec[i];
        vec_new(f);
        vec = vec->v_link2;
    }

    dc = TMALLOC(double, ngood);
    for (i = 0; i < ngood; i++)
        dc[i] = 0;

    for (k = 1; k < tlen; k++) {
        double amp = win[k]/(tlen-1);
        for (i = 0; i < ngood; i++) {
            dc[i] += tdvec[i][k]*amp;
        }
    }
    trace = cp_getvar("spectrace", CP_BOOL, NULL);
    for (j = (startf == 0 ? 1 : 0); j < fpts; j++) {
        freq[j] = startf + j*stepf;
        if (trace)
            fprintf(cp_err, "spec: %e Hz: \r", freq[j]);
        for (i = 0; i < ngood; i++) {
            fdvec[i][j].cx_real = 0;
            fdvec[i][j].cx_imag = 0;
        }
        for (k = 1; k < tlen; k++) {
            double
                amp = 2*win[k]/(tlen-1),
                rad = 2*M_PI*time[k]*freq[j],
                cosa = amp*cos(rad),
                sina = amp*sin(rad);
            for (i = 0; i < ngood; i++) {
                double value = tdvec[i][k]-dc[i];
                fdvec[i][j].cx_real += value*cosa;
                fdvec[i][j].cx_imag += value*sina;
            }
        }
#ifdef HAS_WINDOWS
        SetAnalyse("spec", (int)(j * 1000./ fpts));
#endif
    }
    if (startf == 0) {
        freq[0] = 0;
        for (i = 0; i < ngood; i++) {
            fdvec[i][0].cx_real = dc[i];
            fdvec[i][0].cx_imag = 0;
        }
    }
    if (trace)
        fprintf(cp_err, "                           \r");
    tfree(dc);
    tfree(tdvec);
    tfree(fdvec);

#ifdef KEEPWINDOW
    f = alloc(struct dvec);
    ZERO(f, struct dvec);
    f->v_name = copy("win");
    f->v_type = SV_NOTYPE;
    f->v_flags = (VF_REAL | VF_PERMANENT);
    f->v_length = tlen;
    f->v_realdata = win;
    vec_new(f);
#else
    tfree(win);
#endif

}
