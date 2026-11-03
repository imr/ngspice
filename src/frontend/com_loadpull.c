/* Enhancement-234: the `loadpull` power-amplifier load-pull analysis.
 *
 * Load-pull sweeps the LOAD impedance (Gamma_L) a device/PA output sees over a
 * grid inside the Smith chart and, at each point, runs a large-signal transient,
 * extracts the fundamental via a direct DFT, and reports contours of output
 * power / gain / PAE / drain efficiency. `-source` sweeps the SOURCE impedance
 * (Gamma_s) instead (source-pull). It rides the existing .tran engine and the
 * `alter` mechanism (as com_optimize/com_sweep do); contours render with
 * `pyplot -contour` (Enhancement-218).
 *
 * The load is presented by three series elements R, L, C (out node -> ground):
 * for each Gamma the target Z = z0*(1+G)/(1-G) = R + jX is synthesized by setting
 * the resistor to R and the L/C to give +X (inductor) or -X (capacitor); the
 * unused reactor is parked at a near-short so the branch is AC-coupled R+jX at f0.
 *
 * Verified against analytic max-power transfer: a linear Thevenin source (Vs, Zs)
 * delivers Pmax = |Vs|^2/(8 Rs) into a conjugate-matched load, so Pout must peak
 * at Gamma_L = conj(Gamma_s) with that value.
 */

#include "ngspice/ngspice.h"
#include "ngspice/cpdefs.h"
#include "ngspice/ftedefs.h"
#include "ngspice/dvec.h"
#include "ngspice/wordlist.h"
#include "ngspice/fteext.h"
#include "ngspice/cpextern.h"
#include "ngspice/cktdefs.h"
#include "ngspice/sim.h"

#include "com_loadpull.h"

/* ---- a waveform: time + value samples pulled from the last tran ---- */
typedef struct {
    double *t;
    double *y;
    int n;
} lpwave;

/* Run one command synchronously through the command table (cp_evloop would
 * defer it), exactly as com_stb's stb_run. */
static void lp_run(const char *cmdstr)
{
    wordlist *wl = cp_lexer((char *) cmdstr);
    int i;
    if (!wl || !wl->wl_word) { if (wl) wl_free(wl); return; }
    for (i = 0; cp_coms[i].co_comname; i++)
        if (strcasecmp(cp_coms[i].co_comname, wl->wl_word) == 0)
            break;
    if (cp_coms[i].co_comname && cp_coms[i].co_func)
        cp_coms[i].co_func(wl->wl_next);
    wl_free(wl);
}

/* Evaluate an expression, copy its real data into a fresh array. */
static double *lp_eval(const char *expr, int *lenp)
{
    struct pnode *pn = ft_getpnames_from_string(expr, TRUE);
    double *out = NULL;
    *lenp = 0;
    if (pn) {
        struct dvec *v = ft_evaluate(pn);
        if (v && v->v_length >= 1) {
            int n = v->v_length, i;
            out = TMALLOC(double, n);
            for (i = 0; i < n; i++)
                out[i] = isreal(v) ? v->v_realdata[i] : realpart(v->v_compdata[i]);
            *lenp = n;
        }
        if (v && !pn->pn_value)
            vec_free(v);
        free_pnode(pn);
    }
    return out;
}

/* fundamental phasor of y(t) over the last `nper` periods of `w`:
 *   Y = (2/W) * integral y(t) e^{-j w0 t} dt   (trapezoidal on the tran samples)
 * also returns the DC average of y over the same window in *dc (may be NULL). */
static void lp_fundamental(const lpwave *w, double f0, int nper,
                           double *yre, double *yim, double *dc)
{
    double w0 = 2.0 * M_PI * f0, tp = 1.0 / f0;
    double tend = w->t[w->n - 1];
    double tstart = tend - nper * tp;
    double re = 0.0, im = 0.0, avg = 0.0, wsum = 0.0;
    int i;
    if (tstart < w->t[0])
        tstart = w->t[0];
    for (i = 0; i < w->n; i++) {
        double ti = w->t[i], wt, ph;
        if (ti < tstart)
            continue;
        /* trapezoidal weight = half the span to the neighbours */
        {
            double tlo = (i > 0) ? w->t[i - 1] : ti;
            double thi = (i < w->n - 1) ? w->t[i + 1] : ti;
            if (tlo < tstart) tlo = tstart;
            wt = 0.5 * (thi - tlo);
        }
        ph = w0 * ti;
        re += w->y[i] * cos(ph) * wt;
        im += w->y[i] * sin(ph) * wt;
        avg += w->y[i] * wt;
        wsum += wt;
    }
    if (wsum <= 0.0) wsum = 1.0;
    *yre = 2.0 * re / wsum;
    *yim = -2.0 * im / wsum;
    if (dc)
        *dc = avg / wsum;
}

/* Gamma -> Z = z0 (1+G)/(1-G) */
static void lp_gamma_to_z(double gr, double gi, double z0, double *R, double *X)
{
    double nr = 1.0 + gr, ni = gi;          /* 1+G   */
    double dr = 1.0 - gr, di = -gi;         /* 1-G   */
    double d = dr * dr + di * di;
    double zr = (nr * dr + ni * di) / d;
    double zi = (ni * dr - nr * di) / d;
    *R = z0 * zr;
    *X = z0 * zi;
}

/* Resolve a source's + terminal (node 1) into out[], "" if not found. ngspice
 * stores instance names lowercased, so lowercase before findInstance.  NOTE:
 * do NOT call INPretrieve here -- it can replace the pointer with the interned
 * UID string that the instance's own name field also points at, and freeing
 * that below would be a use-after-free (it corrupts the source's name on any
 * later re-setup).  Top-level device names need no subckt translation. */
static void lp_srcnode(const char *srcname, char *out, size_t outsz)
{
    char *look = copy(srcname);
    GENinstance *inst; CKTnode *nd; IFuid uid;
    char *p;
    out[0] = '\0';
    for (p = look; *p; p++) *p = (char) tolower((unsigned char) *p);
    inst = ft_sim->findInstance(ft_curckt->ci_ckt, look);
    if (inst && CKTinst2Node(ft_curckt->ci_ckt, inst, 1, &nd, &uid) == OK)
        (void) snprintf(out, outsz, "%s", (char *) uid);
    tfree(look);
}

void com_loadpull(wordlist *wl)
{
    char *rname = NULL, *lname = NULL, *cname = NULL;
    char *outnode = NULL, *drive = NULL, *supply = NULL;
    double f0 = 0.0, z0 = 50.0, gmax = 0.85;
    int ngrid = 15, nper = 20, npts = 50;
    int source_mode = 0;
    char cmd[256], expr[256];
    wordlist *w;

    if (!ft_curckt || !ft_curckt->ci_ckt) {
        fprintf(cp_err, "loadpull: no circuit loaded.\n");
        return;
    }

    /* ---- parse flags ---- */
    for (w = wl; w; w = w->wl_next) {
        const char *a = w->wl_word;
        if (eq(a, "-load") && w->wl_next && w->wl_next->wl_next && w->wl_next->wl_next->wl_next) {
            rname = w->wl_next->wl_word;
            lname = w->wl_next->wl_next->wl_word;
            cname = w->wl_next->wl_next->wl_next->wl_word;
            w = w->wl_next->wl_next->wl_next;
        } else if (eq(a, "-out") && w->wl_next) {
            outnode = w->wl_next->wl_word; w = w->wl_next;
        } else if (eq(a, "-drive") && w->wl_next) {
            drive = w->wl_next->wl_word; w = w->wl_next;
        } else if (eq(a, "-supply") && w->wl_next) {
            supply = w->wl_next->wl_word; w = w->wl_next;
        } else if (eq(a, "-f") && w->wl_next) {
            f0 = atof(w->wl_next->wl_word); w = w->wl_next;
        } else if (eq(a, "-z0") && w->wl_next) {
            z0 = atof(w->wl_next->wl_word); w = w->wl_next;
        } else if (eq(a, "-n") && w->wl_next) {
            ngrid = atoi(w->wl_next->wl_word); w = w->wl_next;
        } else if (eq(a, "-gmax") && w->wl_next) {
            gmax = atof(w->wl_next->wl_word); w = w->wl_next;
        } else if (eq(a, "-nper") && w->wl_next) {
            nper = atoi(w->wl_next->wl_word); w = w->wl_next;
        } else if (eq(a, "-npts") && w->wl_next) {
            npts = atoi(w->wl_next->wl_word); w = w->wl_next;
        } else if (eq(a, "-source") && w->wl_next && w->wl_next->wl_next && w->wl_next->wl_next->wl_next) {
            /* source-pull: sweep these source R,L,C instead of the load */
            rname = w->wl_next->wl_word;
            lname = w->wl_next->wl_next->wl_word;
            cname = w->wl_next->wl_next->wl_next->wl_word;
            w = w->wl_next->wl_next->wl_next;
            source_mode = 1;
        }
    }

    if (!rname || !lname || !cname || !outnode || !drive || f0 <= 0.0) {
        fprintf(cp_err,
          "usage: loadpull -load <R> <L> <C> -out <node> -drive <Vsrc> -f <freq>\n"
          "                [-supply <Vsrc>] [-z0 50] [-n 15] [-gmax 0.85]\n"
          "                [-nper 20] [-npts 50] | -source <Rs> <Ls> <Cs> ...\n"
          "  -load    three series R,L,C elements forming the swept load (out->gnd)\n"
          "  -out     output node (fundamental power measured here)\n"
          "  -drive   input drive source (for Pin / gain)\n"
          "  -supply  DC supply source (for PAE / drain efficiency)\n"
          "  -source  sweep these source R,L,C instead of the load (source-pull)\n"
          "Sweeps Gamma over |Gamma|<=gmax and stores gamma_re,gamma_im,pout_dbm,\n"
          "gain_db[,pae,eff] in a 'loadpull' plot; view with `pyplot -contour`.\n");
        return;
    }
    if (gmax >= 1.0) gmax = 0.98;
    if (ngrid < 3) ngrid = 3;

    {
        double tp = 1.0 / f0;
        double tstep = tp / npts;
        double tstop = 2.0 * nper * tp;        /* settle nper, integrate last nper */
        char innode[128] = "", snode[128] = "";
        /* set the circuit up (populates the instance lists) so findInstance
         * resolves the source terminals; then look up the drive + supply nodes
         * once (they do not change over the sweep). A tiny tran (not op) is used
         * so the branch-current output nodes match the sweep's trans exactly. */
        {
            double tp0 = 1.0 / f0;
            (void) snprintf(cmd, sizeof cmd, "tran %.10g %.10g", tp0 / 50.0, tp0);
            lp_run(cmd);
        }
        char drivelc[128], supplylc[128] = "";
        { int q; for (q = 0; drive[q] && q < 127; q++) drivelc[q] = (char) tolower((unsigned char) drive[q]); drivelc[q] = '\0'; }
        if (supply) { int q; for (q = 0; supply[q] && q < 127; q++) supplylc[q] = (char) tolower((unsigned char) supply[q]); supplylc[q] = '\0'; }
        lp_srcnode(drive, innode, sizeof innode);
        if (supply)
            lp_srcnode(supply, snode, sizeof snode);

        /* result buffers (rectangular grid clipped to |G|<=gmax) */
        int cap = (ngrid + 1) * (ngrid + 1), k = 0;
        double *gre = TMALLOC(double, cap), *gim = TMALLOC(double, cap);
        double *poutv = TMALLOC(double, cap), *gainv = TMALLOC(double, cap);
        double *paev = TMALLOC(double, cap), *effv = TMALLOC(double, cap);
        double bestp = -1e30, bestgr = 0, bestgi = 0;
        int ig, jg;

        fprintf(cp_out, "loadpull: %s-pull, f0=%.6g Hz, z0=%g, |Gamma|<=%.3g, "
                "%dx%d grid, %d periods/point...\n",
                source_mode ? "source" : "load", f0, z0, gmax, ngrid, ngrid, nper);

        for (ig = 0; ig <= ngrid; ig++) {
            for (jg = 0; jg <= ngrid; jg++) {
                double gr = -gmax + 2.0 * gmax * ig / ngrid;
                double gi = -gmax + 2.0 * gmax * jg / ngrid;
                double R, X, L, C;
                lpwave vo, ii, vi, is;
                int no = 0, ni = 0, nv = 0, ns = 0, nt = 0;
                double *tt;
                double vore, voim, iire, iiim, vire, viim, idc = 0.0;
                double pout, pin, pdc = 0.0, gain, pae = 0.0, eff = 0.0;

                if (hypot(gr, gi) > gmax + 1e-9)
                    continue;

                lp_gamma_to_z(gr, gi, z0, &R, &X);
                if (R < 1e-6) R = 1e-6;
                if (X >= 0.0) { L = X / (2.0 * M_PI * f0); C = 1e-3; }
                else          { L = 1e-15;                 C = -1.0 / (2.0 * M_PI * f0 * X); }

                (void) snprintf(cmd, sizeof cmd, "alter %s = %.10g", rname, R); lp_run(cmd);
                (void) snprintf(cmd, sizeof cmd, "alter %s = %.10g", lname, L); lp_run(cmd);
                (void) snprintf(cmd, sizeof cmd, "alter %s = %.10g", cname, C); lp_run(cmd);
                (void) snprintf(cmd, sizeof cmd, "tran %.10g %.10g", tstep, tstop); lp_run(cmd);

                tt = lp_eval("time", &nt);
                (void) snprintf(expr, sizeof expr, "v(%s)", outnode);
                vo.y = lp_eval(expr, &no); vo.t = tt; vo.n = nt;
                (void) snprintf(expr, sizeof expr, "%s#branch", drivelc);
                ii.y = lp_eval(expr, &ni); ii.t = tt; ii.n = nt;
                if (innode[0]) {
                    (void) snprintf(expr, sizeof expr, "v(%s)", innode);
                    vi.y = lp_eval(expr, &nv); vi.t = tt; vi.n = nt;
                } else { vi.y = NULL; }
                if (supply) {
                    (void) snprintf(expr, sizeof expr, "%s#branch", supplylc);
                    is.y = lp_eval(expr, &ns); is.t = tt; is.n = nt;
                } else { is.y = NULL; }

                if (!tt || !vo.y || nt < 4 || no != nt) {
                    fprintf(cp_err, "loadpull: tran/extract failed at G=(%.3g,%.3g).\n", gr, gi);
                    tfree(tt); tfree(vo.y); tfree(ii.y); tfree(vi.y); tfree(is.y);
                    continue;
                }

                /* fundamentals */
                lp_fundamental(&vo, f0, nper, &vore, &voim, NULL);
                /* Pout = 0.5 |Vout|^2 Re(1/Z) = 0.5 |Vout|^2 R/|Z|^2 */
                {
                    double vmag2 = vore * vore + voim * voim;
                    double z2 = R * R + X * X;
                    pout = 0.5 * vmag2 * R / z2;
                }
                /* Pin = 0.5 Re(Vin * conj(Iin)); Iin = current from drive into node */
                pin = 0.0;
                if (vi.y && ii.y) {
                    lp_fundamental(&ii, f0, nper, &iire, &iiim, NULL);
                    lp_fundamental(&vi, f0, nper, &vire, &viim, NULL);
                    /* i(Vsrc) flows + -> - internally; power INTO the device = -0.5 Re(V I*) */
                    pin = -0.5 * (vire * iire + viim * iiim);
                    if (pin < 0.0) pin = -pin;   /* orientation-agnostic magnitude */
                }
                /* Pdc = Vdd * Idc(supply) */
                if (is.y && snode[0]) {
                    double dummy_re, dummy_im, vdd = 0.0;
                    lpwave sv; int nsv = 0;
                    lp_fundamental(&is, f0, nper, &dummy_re, &dummy_im, &idc);
                    (void) snprintf(expr, sizeof expr, "v(%s)", snode);
                    sv.y = lp_eval(expr, &nsv); sv.t = tt; sv.n = nt;
                    if (sv.y) {
                        double r_, i_; lp_fundamental(&sv, f0, nper, &r_, &i_, &vdd);
                        tfree(sv.y);
                    }
                    pdc = fabs(vdd * idc);
                }

                gain = (pin > 0.0) ? 10.0 * log10(pout / pin) : 0.0;
                if (pdc > 0.0) {
                    pae = 100.0 * (pout - pin) / pdc;
                    eff = 100.0 * pout / pdc;
                }

                gre[k] = gr; gim[k] = gi;
                poutv[k] = 10.0 * log10(pout / 1e-3);   /* dBm */
                gainv[k] = gain; paev[k] = pae; effv[k] = eff;
                if (pout > bestp) { bestp = pout; bestgr = gr; bestgi = gi; }
                k++;

                tfree(tt); tfree(vo.y); tfree(ii.y); tfree(vi.y); tfree(is.y);
            }
        }

        /* restore something sane on the load (last set values are fine) */

        if (k < 3) {
            fprintf(cp_err, "loadpull: too few valid points.\n");
        } else {
            struct plot *pl = plot_alloc("loadpull");
            struct dvec *d;
            int i;
            pl->pl_name  = copy(source_mode ? "Source-pull" : "Load-pull");
            pl->pl_title = copy(ft_curckt->ci_name ? ft_curckt->ci_name : "loadpull");
            plot_new(pl);
            plot_setcur(pl->pl_typename);
#define LP_VEC(nm, arr) \
            d = dvec_alloc(copy(nm), SV_NOTYPE, (short)(VF_REAL|VF_PERMANENT), k, NULL); \
            for (i = 0; i < k; i++) d->v_realdata[i] = (arr)[i]; vec_new(d);
            LP_VEC("gamma_re", gre)      /* first permanent -> scale */
            LP_VEC("gamma_im", gim)
            LP_VEC("pout_dbm", poutv)
            LP_VEC("gain_db",  gainv)
            if (supply) { LP_VEC("pae", paev) LP_VEC("eff", effv) }
#undef LP_VEC
            {
                double br, bx;
                lp_gamma_to_z(bestgr, bestgi, z0, &br, &bx);
                fprintf(cp_out, "\n%s-pull result (%d points):\n",
                        source_mode ? "Source" : "Load", k);
                fprintf(cp_out, "  optimum Gamma  = %.4f angle %.2f deg  (Z = %.3g %+.3g j ohm)\n",
                        hypot(bestgr, bestgi), atan2(bestgi, bestgr) * 180.0 / M_PI, br, bx);
                fprintf(cp_out, "  peak Pout      = %.4f dBm\n", 10.0 * log10(bestp / 1e-3));
                fprintf(cp_out, "  -> stored gamma_re,gamma_im,pout_dbm,gain_db%s in 'loadpull';\n",
                        supply ? ",pae,eff" : "");
                fprintf(cp_out, "     `pyplot -contour gamma_re gamma_im pout_dbm`\n");
            }
        }
        tfree(gre); tfree(gim); tfree(poutv); tfree(gainv); tfree(paev); tfree(effv);
    }
}
