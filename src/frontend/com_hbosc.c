/**********
Enhancement-140: autonomous harmonic balance for oscillators + phase noise.

  hbosc <oscnode> <K> [fguess] [tstab]   -- autonomous HB: find the oscillator's
                                            steady state (harmonics + frequency)
  phasenoise <fstart> <fstop> [points]   -- the phase-noise spectrum L(df)

`hbosc` runs a short transient (from the deck's .ic) to seed the limit cycle, estimates
the oscillation frequency and amplitude, and hands them to HBOSCanalyze() which refines
(V, w0) by a bordered Newton and retains the operating point. `phasenoise` then extracts
the perturbation projection vector (PPV) and folds the device noise to L(df). The engines
are in spicelib/analysis/dcpss.c.
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cpdefs.h"
#include "ngspice/cktdefs.h"
#include "ngspice/ftedefs.h"
#include "ngspice/dvec.h"
#include "ngspice/fteext.h"
#include "ngspice/wordlist.h"
#include "ngspice/cpextern.h"

#include "circuits.h"
#include "com_hbosc.h"

static void hbosc_run_cmd(const char *cmdstr)
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

static double hboscnum(const char *w)
{
    char *s = (char *) w;
    double v = 0.0;
    if (ft_numparse(&s, FALSE, &v) < 0)
        v = atof(w);
    return v;
}

static int hbosc_node(CKTcircuit *ckt, const char *name)
{
    int numNames = 0, i, num = 0;
    IFuid *nameList = NULL;
    if (CKTnames(ckt, &numNames, &nameList) != OK || !nameList)
        return 0;
    for (i = 0; i < numNames; i++)
        if (nameList[i] && strcmp((const char *) nameList[i], name) == 0) { num = i + 1; break; }
    tfree(nameList);
    return num;
}

void
com_hbosc(wordlist *wl)
{
    CKTcircuit *ckt;
    const char *oscname;
    double fguess = 0.0, tstab = 0.0, ampseed, f0est;
    int    oscNode, K, verbose, err;
    char   cmd[128];
    struct pnode *pn;
    struct dvec  *v, *sc;
    double *tt, *vv;
    int    n, i0, i, ncross;
    double vmax, tlast, tprev;

    if (!ft_curckt || !ft_curckt->ci_ckt) {
        fprintf(cp_err, "Error: hbosc: there is no circuit loaded.\n");
        return;
    }
    ckt = ft_curckt->ci_ckt;
    if (!wl || !wl->wl_next) {
        fprintf(cp_err, "Usage: hbosc <oscnode> <K> [fguess] [tstab]   (the deck needs a .ic to start the oscillation)\n");
        return;
    }
    oscname = wl->wl_word;
    K = (int) hboscnum(wl->wl_next->wl_word);
    if (wl->wl_next->wl_next) {
        fguess = hboscnum(wl->wl_next->wl_next->wl_word);
        if (wl->wl_next->wl_next->wl_next)
            tstab = hboscnum(wl->wl_next->wl_next->wl_next->wl_word);
    }
    if (K < 1) { fprintf(cp_err, "Error: hbosc: need K >= 1.\n"); return; }
    if (fguess <= 0.0) fguess = 1e6;                 /* a default if none given */
    if (tstab <= 0.0) tstab = 300.0 / fguess;        /* ~300 periods to settle */

    /* build up the limit cycle with a transient (uic honours the deck's .ic) */
    (void) snprintf(cmd, sizeof cmd, "tran %.6g %.6g uic", 1.0/(fguess*40.0), tstab);
    hbosc_run_cmd(cmd);

    /* fetch v(oscnode) + its time scale */
    (void) snprintf(cmd, sizeof cmd, "v(%s)", oscname);
    pn = ft_getpnames_from_string(cmd, TRUE);
    if (!pn) { fprintf(cp_err, "Error: hbosc: cannot read v(%s).\n", oscname); return; }
    v = ft_evaluate(pn);
    sc = (v && v->v_scale) ? v->v_scale : vec_get("time");
    if (!v || !isreal(v) || v->v_length < 8 || !sc || sc->v_length < v->v_length) {
        fprintf(cp_err, "Error: hbosc: no usable transient for v(%s).\n", oscname);
        if (pn && !pn->pn_value && v) vec_free(v);
        if (pn) free_pnode(pn);
        return;
    }
    tt = sc->v_realdata; vv = v->v_realdata; n = v->v_length;

    /* estimate the oscillation: amplitude = max|v| over the last third; frequency from
     * the mean spacing of upward zero crossings there. */
    i0 = (2*n)/3;
    vmax = 0.0;
    for (i = i0; i < n; i++) if (fabs(vv[i]) > vmax) vmax = fabs(vv[i]);
    ncross = 0; tlast = tprev = 0.0;
    for (i = i0 + 1; i < n; i++)
        if (vv[i-1] <= 0.0 && vv[i] > 0.0) {         /* upward crossing */
            double tc = tt[i-1] + (tt[i]-tt[i-1]) * (-vv[i-1])/(vv[i]-vv[i-1]);
            if (ncross == 0) tprev = tc;
            tlast = tc; ncross++;
        }
    f0est = (ncross > 1 && tlast > tprev) ? (double)(ncross-1)/(tlast-tprev) : fguess;

    ampseed = vmax;
    oscNode = hbosc_node(ckt, oscname);
    if (pn && !pn->pn_value && v) vec_free(v);
    if (pn) free_pnode(pn);
    if (oscNode <= 0) { fprintf(cp_err, "Error: hbosc: unknown oscillator node '%s'.\n", oscname); return; }
    if (vmax < 1e-9) {
        fprintf(cp_err, "Error: hbosc: no oscillation detected (add a `.ic` to start it).\n");
        return;
    }

    verbose = cp_getvar("hbosc_verbose", CP_BOOL, NULL, 0);
    err = HBOSCanalyze(ckt, oscNode, K, 0, f0est, ampseed, 60, 1e-11, verbose ? 1 : 0);
    if (err != OK)
        fprintf(cp_err, "hbosc: autonomous harmonic balance did not complete (error %d).\n", err);
}

void
com_phasenoise(wordlist *wl)
{
    CKTcircuit *ckt;
    double fstart, fstop;
    int    npts = 21, verbose, err;

    if (!ft_curckt || !ft_curckt->ci_ckt) {
        fprintf(cp_err, "Error: phasenoise: there is no circuit loaded.\n");
        return;
    }
    ckt = ft_curckt->ci_ckt;
    if (!wl || !wl->wl_next) {
        fprintf(cp_err, "Usage: phasenoise <fstart> <fstop> [points]   (run `hbosc` first)\n");
        return;
    }
    fstart = hboscnum(wl->wl_word);
    fstop  = hboscnum(wl->wl_next->wl_word);
    if (wl->wl_next->wl_next)
        npts = (int) hboscnum(wl->wl_next->wl_next->wl_word);
    if (fstart <= 0.0 || fstop < fstart) {
        fprintf(cp_err, "Error: phasenoise: need 0 < fstart <= fstop.\n");
        return;
    }
    if (npts < 1) npts = 1;

    verbose = cp_getvar("phasenoise_verbose", CP_BOOL, NULL, 0);
    err = PhaseNoiseAnalyze(ckt, fstart, fstop, npts, verbose ? 1 : 0);
    if (err != OK)
        fprintf(cp_err, "phasenoise: did not complete (error %d).\n", err);
}
