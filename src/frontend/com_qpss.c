/**********
Enhancement-133: quasi-periodic steady state (QPSS) for two commensurate tones.

`qpss <expr> <f1> <f2> [periods] [maxorder]`

A two-tone / multi-fundamental steady-state analysis. For two commensurate tones
f1 and f2 (a rational ratio, so they share a beat frequency fb = gcd(f1,f2)) the
circuit's steady state is periodic at fb; QPSS runs a transient over a few beat
periods to reach that steady state, then resolves the response into the two-tone
spectrum -- every mixing product k1*f1 + k2*f2, including the intermodulation
distortion (IM3 at 2f1-f2 / 2f2-f1, etc.) that a single-tone analysis cannot show.

Rather than a beat-frequency shooting PSS (which is slow and needs reactive state),
QPSS uses the robust transient-sampling method: run the transient, then take the
LAST beat period and evaluate the Fourier coefficient **directly at each exact
intermod frequency** k1*f1 + k2*f2 (a direct DFT, exact for commensurate tones --
no resampling or FFT-bin rounding). Each product is labelled by its 2-D harmonic
index (k1, k2), the defining QPSS output.

Independent of the linear solver (it drives an ordinary transient).

Enhancement-136: `qpss <expr> <f1> <f2> hb [K1] [K2]` selects a frequency-domain
two-tone Harmonic Balance engine instead of the transient path -- the TRUE
quasi-periodic steady state, which works for INCOMMENSURATE tones (no common beat
period) and retains its operating point for the small-signal `qpac` (E-137). The
engine (QPSShb, spicelib/analysis/dcpss.c) samples the devices on a 2-D phase grid,
2-D DFTs to the conversion matrix, and Newton-solves in the frequency domain.
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cpdefs.h"
#include "ngspice/cktdefs.h"
#include "ngspice/ftedefs.h"
#include "ngspice/dvec.h"
#include "ngspice/wordlist.h"
#include "ngspice/fteext.h"
#include "ngspice/cpextern.h"

#include "circuits.h"
#include "com_qpss.h"

/* Run one command synchronously through the command table (see com_optimize.c):
 * cp_evloop() called re-entrantly would defer it to the outer loop. */
static void qpss_run_cmd(const char *cmdstr)
{
    wordlist *wl = cp_lexer((char *) cmdstr);
    int i;

    if (!wl || !wl->wl_word) {
        if (wl) wl_free(wl);
        return;
    }
    for (i = 0; cp_coms[i].co_comname; i++)
        if (strcasecmp(cp_coms[i].co_comname, wl->wl_word) == 0)
            break;
    if (cp_coms[i].co_comname && cp_coms[i].co_func)
        cp_coms[i].co_func(wl->wl_next);
    wl_free(wl);
}

/* SPICE-style number (k / meg / u / n / p ...). */
static double qpssnum(const char *w)
{
    char *s = (char *) w;
    double v = 0.0;
    if (ft_numparse(&s, FALSE, &v) < 0)
        v = atof(w);
    return v;
}

/* Greatest common "divisor" of two frequencies, by the Euclidean algorithm on
 * reals with a relative tolerance -- the beat frequency of two commensurate tones. */
static double real_gcd(double a, double b)
{
    double tol;
    a = fabs(a);
    b = fabs(b);
    tol = (a < b ? a : b) * 1e-6;
    if (tol <= 0.0)
        return 0.0;
    while (b > tol) {
        double t = fmod(a, b);
        a = b;
        b = t;
    }
    return a;
}


void
com_qpss(wordlist *wl)
{
    const char *expr;
    double f1, f2, fb, fmax, tstop, tstep, T, wstart, wend;
    int    periods = 8, maxorder = 5;
    int    n, k1, k2, i0, ord, M = 0;   /* M: uniform-resample length (Enhancement-319) */
    char   cmd[256];
    struct pnode *pn;
    struct dvec  *v, *sc;
    double *tt, *vv, *ur = NULL;   /* ur: last period resampled onto a uniform grid (E-319) */

    if (!ft_curckt || !ft_curckt->ci_ckt) {
        fprintf(cp_err, "Error: qpss: there is no circuit loaded.\n");
        return;
    }
    if (!wl || !wl->wl_next || !wl->wl_next->wl_next) {
        fprintf(cp_err, "Usage: qpss <expr> <f1> <f2> [periods] [maxorder]\n");
        return;
    }

    expr = wl->wl_word;
    f1 = qpssnum(wl->wl_next->wl_word);
    f2 = qpssnum(wl->wl_next->wl_next->wl_word);

    /* Enhancement-136: `qpss <expr> <f1> <f2> hb [K1] [K2]` selects the
     * frequency-domain two-tone Harmonic Balance engine -- the true,
     * incommensurate-capable quasi-periodic steady state, which also retains
     * the operating point for `qpac` -- instead of the E-133 transient path.
     * The output is a per-node two-tone spectrum, so `expr` is not used here. */
    {
        wordlist *w;
        for (w = wl->wl_next->wl_next->wl_next; w; w = w->wl_next)
            if (strcasecmp(w->wl_word, "hb") == 0) {
                CKTcircuit *ckt = ft_curckt->ci_ckt;
                int K1 = 3, K2 = 3, verbose, err;
                if (f1 <= 0.0 || f2 <= 0.0 || f1 == f2) {
                    fprintf(cp_err, "Error: qpss hb: need two distinct positive tone "
                                    "frequencies.\n");
                    return;
                }
                if (w->wl_next) {
                    K1 = (int) qpssnum(w->wl_next->wl_word);
                    K2 = w->wl_next->wl_next
                         ? (int) qpssnum(w->wl_next->wl_next->wl_word) : K1;
                }
                if (K1 < 1) K1 = 1;
                if (K2 < 1) K2 = 1;
#ifdef KLU
                if (ft_curckt->ci_defTask &&
                    (ckt->CKTmatrix == NULL || SMPmatSize(ckt->CKTmatrix) <= 0))
                    ckt->CKTkluMODE = ft_curckt->ci_defTask->TSKkluMODE;
#endif
                if (ckt->CKTmatrix == NULL || SMPmatSize(ckt->CKTmatrix) <= 0) {
                    if ((err = CKTsetup(ckt)) != OK || (err = CKTtemp(ckt)) != OK) {
                        fprintf(cp_err, "Error: qpss hb: circuit setup failed.\n");
                        return;
                    }
                }
                verbose = cp_getvar("qpss_verbose", CP_BOOL, NULL, 0);
                ft_curckt->ci_curTask = ft_curckt->ci_defTask;
                ckt->CKTcurJob = ft_curckt->ci_defTask ? ft_curckt->ci_defTask->jobs : NULL;
                err = QPSShb(ckt, f1, f2, K1, K2, 0, 0, 60, 1e-10, verbose ? 1 : 0);
                if (err != OK)
                    fprintf(cp_err, "qpss hb: harmonic balance did not complete "
                                    "(error %d).\n", err);
                return;
            }
    }

    if (wl->wl_next->wl_next->wl_next) {
        periods = (int) qpssnum(wl->wl_next->wl_next->wl_next->wl_word);
        if (wl->wl_next->wl_next->wl_next->wl_next)
            maxorder = (int) qpssnum(wl->wl_next->wl_next->wl_next->wl_next->wl_word);
    }
    if (f1 <= 0.0 || f2 <= 0.0 || f1 == f2) {
        fprintf(cp_err, "Error: qpss: need two distinct positive tone frequencies.\n");
        return;
    }
    if (periods < 1) periods = 1;
    if (maxorder < 1) maxorder = 1;

    fb = real_gcd(f1, f2);
    if (fb <= 0.0) {
        fprintf(cp_err, "Error: qpss: tones f1=%g and f2=%g are not commensurate "
                        "(no common beat frequency).\n", f1, f2);
        return;
    }
    fmax = (f1 > f2 ? f1 : f2);
    T = 1.0 / fb;
    tstop = periods * T;
    /* resolve the highest reported harmonic with ~20 samples per period */
    tstep = 1.0 / (fmax * (maxorder + 1) * 20.0);

    /* run the two-tone transient (uic off; the beat-period settling reaches the
     * periodic steady state).  tmax = tstep keeps the sampling fine and even. */
    (void) snprintf(cmd, sizeof cmd, "tran %.10g %.10g 0 %.10g", tstep, tstop, tstep);
    qpss_run_cmd(cmd);

    /* fetch the output waveform + its time scale */
    pn = ft_getpnames_from_string(expr, TRUE);
    if (!pn) {
        fprintf(cp_err, "Error: qpss: cannot parse output expression '%s'.\n", expr);
        return;
    }
    v = ft_evaluate(pn);
    /* the time axis: an expression temporary drops its scale, so fall back to the
     * current plot's `time` reference vector. */
    sc = (v && v->v_scale) ? v->v_scale : vec_get("time");
    if (!v || !isreal(v) || v->v_length < 4 || !sc || sc->v_length < v->v_length) {
        fprintf(cp_err, "Error: qpss: '%s' produced no usable transient waveform.\n", expr);
        if (pn && !pn->pn_value && v) vec_free(v);
        if (pn) free_pnode(pn);
        return;
    }
    tt = sc->v_realdata;
    vv = v->v_realdata;
    n  = v->v_length;

    /* last beat period window [t_end - T, t_end] */
    wend   = tt[n - 1];
    wstart = wend - T;
    for (i0 = n - 1; i0 > 0 && tt[i0 - 1] >= wstart; i0--)
        ;

    fprintf(cp_out,
            "\nQPSS: two-tone steady state of %s\n"
            "  f1 = %g Hz, f2 = %g Hz, beat fb = %g Hz; %d beat periods, order <= %d\n"
            "  (k1,k2)      frequency [Hz]        |value|         phase [deg]\n",
            expr, f1, f2, fb, periods, maxorder);

    /* Enhancement-319: resample the last beat period onto a UNIFORM grid over EXACTLY
     * [wend - T, wend) and Fourier-project below with the rectangular rule. For commensurate
     * tones every reported harmonic k1*f1 + k2*f2 = m*fb completes an integer m cycles in T,
     * so a uniform DFT over exactly T is EXACT -- a linear circuit's mixing products come out
     * at machine zero. The earlier code integrated (trapezoidally) over the raw transient grid,
     * whose window length was not exactly T, whose steps were non-uniform, and whose endpoints
     * were non-periodic, so it leaked the DC/fundamental (~tstep/T) into every mixing bin (a
     * ~5.8e-4 * |dominant line| floor on a linear two-tone RC where every product must be 0). */
    {
        double num_bins = maxorder * fmax / fb;   /* highest reported harmonic, in fb bins */
        int j, m, target;
        /* Resolve the highest harmonic AND do not downsample the transient grid (which
         * already carries the last period at the run's tstep) -- upsampling never hurts,
         * downsampling would alias and coarsen the interpolation. */
        target = (int) (8.0 * num_bins);
        if (target < n - i0)
            target = n - i0;
        for (M = 64; M < target && M < 65536; M <<= 1)
            ;
        ur = TMALLOC(double, M);
        wstart = wend - T;
        j = (i0 > 0) ? i0 - 1 : 0;                 /* interpolation cursor into (tt, vv) */
        for (m = 0; m < M; m++) {
            double tq = wstart + ((double) m * T) / M;
            while (j + 1 < n && tt[j + 1] < tq)
                j++;
            if (j + 1 >= n) {
                ur[m] = vv[n - 1];
            } else {
                double dtj = tt[j + 1] - tt[j];
                ur[m] = (dtj > 0.0)
                    ? vv[j] + (vv[j + 1] - vv[j]) * (tq - tt[j]) / dtj
                    : vv[j];
            }
        }
    }

    /* Enumerate distinct 2-D harmonics k1*f1 + k2*f2 >= 0 with |k1|+|k2| <= order,
     * and evaluate the Fourier coefficient at each frequency over the uniform last period. */
    for (ord = 0; ord <= maxorder; ord++) {    /* report in ascending total order */
        for (k1 = -maxorder; k1 <= maxorder; k1++) {
            for (k2 = -maxorder; k2 <= maxorder; k2++) {
                double f, w, cre, cim, mag, phase;
                if (abs(k1) + abs(k2) != ord)
                    continue;
                f = k1 * f1 + k2 * f2;
                if (f < -0.5 * fb)             /* keep f >= 0 (real signal: |c(-f)|=|c(f)|) */
                    continue;
                if (f < 0.0) f = 0.0;
                /* skip a product whose frequency duplicates a lower-order one */
                {
                    int dup = 0, a1, a2;
                    for (a1 = -maxorder; a1 <= maxorder && !dup; a1++)
                        for (a2 = -maxorder; a2 <= maxorder; a2++)
                            if ((abs(a1) + abs(a2) < ord) &&
                                fabs(a1 * f1 + a2 * f2 - f) < 0.25 * fb) { dup = 1; break; }
                    if (dup) continue;
                }

                /* Enhancement-319: Fourier coefficient at f = integral of v(t)*exp(-j2pi f t) dt
                 * over exactly one period, rectangular rule on the uniform resampled grid. For a
                 * periodic signal over a full period the rectangular rule equals the trapezoidal
                 * rule (equal endpoints) and, for commensurate f, is exact. */
                cre = cim = 0.0;
                {
                    int m;
                    for (m = 0; m < M; m++) {
                        double ph = 2.0 * M_PI * f * (wstart + ((double) m * T) / M);
                        cre += ur[m] * cos(ph);
                        cim -= ur[m] * sin(ph);
                    }
                    cre *= T / M;
                    cim *= T / M;
                }
                w   = (f < 0.5 * fb) ? (1.0 / T) : (2.0 / T);   /* DC single-sided */
                mag = w * hypot(cre, cim);
                phase = (f < 0.5 * fb) ? 0.0 : atan2(cim, cre) * 180.0 / M_PI;

                fprintf(cp_out, "  (%2d,%2d)   %16.6e   %14.6e   %10.3f\n",
                        k1, k2, f, mag, phase);
            }
        }
    }

    tfree(ur);   /* Enhancement-319 */
    if (pn && !pn->pn_value && v)
        vec_free(v);
    if (pn)
        free_pnode(pn);
}
