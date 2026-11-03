/* Enhancement-253: the `rfstab` two-port stability / gain figure-of-merit report.
 *
 * After a `.sp` analysis (which publishes the scattering parameters as the
 * complex vectors S_1_1, S_1_2, S_2_1, S_2_2 versus `frequency`), `rfstab`
 * post-processes them into the standard linear-two-port RF design metrics, one
 * value per frequency point:
 *
 *   determinant   D  = S11*S22 - S12*S21
 *   Rollett       K  = (1 - |S11|^2 - |S22|^2 + |D|^2) / (2*|S12*S21|)
 *   mu-factor     mu = (1 - |S11|^2) / (|S22 - D*conj(S11)| + |S12*S21|)   (load)
 *   mu'-factor    mu'= (1 - |S22|^2) / (|S11 - D*conj(S22)| + |S12*S21|)   (source)
 *   max stable    MSG = |S21|/|S12|                       (power gain, K<=1)
 *   max available MAG = |S21|/|S12| * (K - sqrt(K^2-1))   (power gain, K>1)
 *
 * A two-port is UNCONDITIONALLY STABLE at a frequency iff K > 1 and |D| < 1,
 * equivalently mu > 1 (and mu' > 1). The results are stored as real vectors
 * (k, magdelta, mu, mu_src, gmax, msg, stable) versus `frequency` in a fresh
 * `rfstab` plot, and a summary (stability verdict, worst-case K/mu, gain range)
 * is printed. It only reads vectors, so it is analysis- and solver-independent.
 *
 *   usage:  rfstab [S11 S12 S21 S22]
 * with no arguments the .sp defaults S_1_1 S_1_2 S_2_1 S_2_2 are used; the four
 * optional vector names allow other sources (e.g. a Touchstone plot).
 */

#include "ngspice/ngspice.h"
#include "ngspice/cpdefs.h"
#include "ngspice/ftedefs.h"
#include "ngspice/dvec.h"
#include "ngspice/wordlist.h"
#include "ngspice/fteext.h"
#include "ngspice/cpextern.h"

#include "com_rfstab.h"

/* ---- small complex helpers on ngcomplex_t ---- */
static ngcomplex_t rfcx(double re, double im) { ngcomplex_t r; r.cx_real = re; r.cx_imag = im; return r; }
static ngcomplex_t rfsub(ngcomplex_t a, ngcomplex_t b) { return rfcx(a.cx_real - b.cx_real, a.cx_imag - b.cx_imag); }
static ngcomplex_t rfmul(ngcomplex_t a, ngcomplex_t b)
{ return rfcx(a.cx_real * b.cx_real - a.cx_imag * b.cx_imag, a.cx_real * b.cx_imag + a.cx_imag * b.cx_real); }
static ngcomplex_t rfconj(ngcomplex_t a) { return rfcx(a.cx_real, -a.cx_imag); }
static double rfabs(ngcomplex_t a) { return hypot(a.cx_real, a.cx_imag); }

/* Evaluate an expression into a fresh ngcomplex_t array (real promoted to
 * complex). Returns NULL and *lenp = 0 on failure. */
static ngcomplex_t *rf_eval(const char *expr, int *lenp)
{
    struct pnode *pn = ft_getpnames_from_string(expr, TRUE);
    ngcomplex_t *out = NULL;
    *lenp = 0;
    if (pn) {
        struct dvec *v = ft_evaluate(pn);
        if (v && v->v_length >= 1) {
            int n = v->v_length, i;
            out = TMALLOC(ngcomplex_t, n);
            for (i = 0; i < n; i++) {
                if (isreal(v)) { out[i] = rfcx(v->v_realdata[i], 0.0); }
                else           { out[i] = v->v_compdata[i]; }
            }
            *lenp = n;
        }
        if (v && !pn->pn_value)
            vec_free(v);
        free_pnode(pn);
    }
    return out;
}

static struct dvec *rf_store(const char *name, int type, double *data, int n)
{
    struct dvec *dv = dvec_alloc(copy(name), type,
                                 (short) (VF_REAL | VF_PERMANENT), n, NULL);
    int i;
    for (i = 0; i < n; i++)
        dv->v_realdata[i] = data[i];
    vec_new(dv);
    return dv;
}

void com_rfstab(wordlist *wl)
{
    const char *e11 = "S_1_1", *e12 = "S_1_2", *e21 = "S_2_1", *e22 = "S_2_2";
    ngcomplex_t *s11 = NULL, *s12 = NULL, *s21 = NULL, *s22 = NULL, *freqc = NULL;
    double *k = NULL, *magd = NULL, *mu = NULL, *mus = NULL, *gmax = NULL,
           *msg = NULL, *stab = NULL, *fr = NULL;
    int n11 = 0, n12 = 0, n21 = 0, n22 = 0, nf = 0, n, i;
    int nstable = 0;
    double kmin = 0, mumin = 0, dmax = 0, gmn = 0, gmx = 0;

    /* optional S-parameter vector names (all four, or none) */
    if (wl && wl->wl_word) {
        if (wl->wl_next && wl->wl_next->wl_next && wl->wl_next->wl_next->wl_next) {
            e11 = wl->wl_word;
            e12 = wl->wl_next->wl_word;
            e21 = wl->wl_next->wl_next->wl_word;
            e22 = wl->wl_next->wl_next->wl_next->wl_word;
        } else {
            fprintf(cp_err, "usage: rfstab [S11 S12 S21 S22]\n"
                    "  with no args, the .sp defaults S_1_1 S_1_2 S_2_1 S_2_2 "
                    "are used.\n");
            return;
        }
    }

    s11 = rf_eval(e11, &n11);
    s12 = rf_eval(e12, &n12);
    s21 = rf_eval(e21, &n21);
    s22 = rf_eval(e22, &n22);
    freqc = rf_eval("frequency", &nf);

    if (!s11 || !s12 || !s21 || !s22) {
        fprintf(cp_err, "rfstab: could not read the S-parameters (%s ...). "
                "Run a `.sp` analysis first, or pass the four vector names.\n",
                e11);
        goto done;
    }
    n = n11;
    if (n12 < n) n = n12;
    if (n21 < n) n = n21;
    if (n22 < n) n = n22;
    if (n < 1) { fprintf(cp_err, "rfstab: empty S-parameter vectors.\n"); goto done; }

    k    = TMALLOC(double, n);  magd = TMALLOC(double, n);
    mu   = TMALLOC(double, n);  mus  = TMALLOC(double, n);
    gmax = TMALLOC(double, n);  msg  = TMALLOC(double, n);
    stab = TMALLOC(double, n);  fr   = TMALLOC(double, n);

    for (i = 0; i < n; i++) {
        ngcomplex_t S11 = s11[i], S12 = s12[i], S21 = s21[i], S22 = s22[i];
        ngcomplex_t D = rfsub(rfmul(S11, S22), rfmul(S12, S21));   /* S11S22-S12S21 */
        double aS11 = rfabs(S11), aS22 = rfabs(S22);
        double aD   = rfabs(D);
        double a12  = rfabs(S12), a21 = rfabs(S21);
        double p    = a12 * a21;                                  /* |S12*S21| */
        double Kv;

        magd[i] = aD;
        fr[i]   = (i < nf) ? freqc[i].cx_real : (double) i;

        Kv = (p > 0.0)
             ? (1.0 - aS11 * aS11 - aS22 * aS22 + aD * aD) / (2.0 * p)
             : 1.0e30;
        k[i] = Kv;

        /* mu (load) and mu' (source) stability factors */
        {
            double dl = rfabs(rfsub(S22, rfmul(D, rfconj(S11)))) + p;
            double ds = rfabs(rfsub(S11, rfmul(D, rfconj(S22)))) + p;
            mu[i]  = (dl > 0.0) ? (1.0 - aS11 * aS11) / dl : 1.0e30;
            mus[i] = (ds > 0.0) ? (1.0 - aS22 * aS22) / ds : 1.0e30;
        }

        /* MSG = |S21/S12|; MAG = MSG*(K - sqrt(K^2-1)) for K>1 (power gains, dB) */
        msg[i] = (a12 > 0.0) ? 10.0 * log10(a21 / a12) : 1.0e30;
        if (Kv > 1.0 && a12 > 0.0) {
            double g = (a21 / a12) * (Kv - sqrt(Kv * Kv - 1.0));
            gmax[i] = 10.0 * log10(g);
        } else {
            gmax[i] = msg[i];               /* K<=1: only the max stable gain */
        }

        stab[i] = (Kv > 1.0 && aD < 1.0) ? 1.0 : 0.0;
        if (stab[i] > 0.5) nstable++;

        if (i == 0 || Kv < kmin) kmin = Kv;
        if (i == 0 || mu[i] < mumin) mumin = mu[i];
        if (i == 0 || aD > dmax) dmax = aD;
        if (i == 0 || gmax[i] < gmn) gmn = gmax[i];
        if (i == 0 || gmax[i] > gmx) gmx = gmax[i];
    }

    /* store results in a fresh `rfstab` plot */
    {
        struct plot *pl = plot_alloc("rfstab");
        struct dvec *sc;
        pl->pl_name  = copy("RF two-port stability");
        pl->pl_title = copy(ft_curckt && ft_curckt->ci_name ? ft_curckt->ci_name
                                                            : "rfstab");
        plot_new(pl);
        plot_setcur(pl->pl_typename);
        sc = dvec_alloc(copy("frequency"), SV_FREQUENCY,
                        (short) (VF_REAL | VF_PERMANENT), n, NULL);
        for (i = 0; i < n; i++) sc->v_realdata[i] = fr[i];
        vec_new(sc);                                    /* first permanent -> scale */
        rf_store("k",        SV_NOTYPE, k,    n);
        rf_store("magdelta", SV_NOTYPE, magd, n);
        rf_store("mu",       SV_NOTYPE, mu,   n);
        rf_store("mu_src",   SV_NOTYPE, mus,  n);
        rf_store("gmax",     SV_NOTYPE, gmax, n);       /* dB */
        rf_store("msg",      SV_NOTYPE, msg,  n);       /* dB */
        rf_store("stable",   SV_NOTYPE, stab, n);
    }

    /* report */
    fprintf(cp_out, "\nRF two-port stability (%d frequency points):\n", n);
    if (nstable == n)
        fprintf(cp_out, "  unconditionally stable at ALL points "
                "(K > 1 and |Delta| < 1 everywhere).\n");
    else
        fprintf(cp_out, "  potentially UNSTABLE at %d of %d points "
                "(K <= 1 or |Delta| >= 1).\n", n - nstable, n);
    fprintf(cp_out, "  worst-case K   = %.4g\n", kmin);
    fprintf(cp_out, "  worst-case mu  = %.4g   (unconditionally stable iff > 1)\n",
            mumin);
    fprintf(cp_out, "  max |Delta|    = %.4g\n", dmax);
    fprintf(cp_out, "  max gain (%s)  = %.4g .. %.4g dB\n",
            nstable == n ? "MAG" : "MAG/MSG", gmn, gmx);
    fprintf(cp_out, "  -> stored k, magdelta, mu, mu_src, gmax, msg, stable in the "
            "'rfstab' plot; `plot k mu` / `plot gmax msg`.\n");

done:
    tfree(s11); tfree(s12); tfree(s21); tfree(s22); tfree(freqc);
    tfree(k); tfree(magd); tfree(mu); tfree(mus);
    tfree(gmax); tfree(msg); tfree(stab); tfree(fr);
}
