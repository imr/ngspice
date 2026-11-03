/**********
Enhancement-137: two-tone small-signal QPAC (quasi-periodic AC) -- `qpac <f_in>`.

Injects a small signal at frequency f_in around the QPSS operating point retained by a
prior `qpss <expr> <f1> <f2> hb`, and reports the response at every sideband
f_in + k1*f1 + k2*f2 -- the two-tone analogue of PAC. The quasi-periodic operating point
mixes the small signal to the sidebands through the same 2-D conversion matrix the QPSS
Newton used as its Jacobian. The heavy lifting is QPACanalyze() (spicelib/analysis/
dcpss.c); this command parses f_in and runs it.
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cpdefs.h"
#include "ngspice/cktdefs.h"
#include "ngspice/ftedefs.h"
#include "ngspice/fteext.h"
#include "ngspice/wordlist.h"
#include "ngspice/cpextern.h"
#include "ngspice/dvec.h"
#include "ngspice/sim.h"

#include "com_qpac.h"

static double qpacnum(const char *w)
{
    char *s = (char *) w;
    double v = 0.0;
    if (ft_numparse(&s, FALSE, &v) < 0)
        v = atof(w);
    return v;
}

/* dec/oct/lin -> 1/2/0 (a frequency sweep), else -1 (single frequency) */
int qp_steptype(const char *w)
{
    if (!strcasecmp(w, "dec")) return 1;
    if (!strcasecmp(w, "oct")) return 2;
    if (!strcasecmp(w, "lin")) return 0;
    return -1;
}

/* upper bound on the number of sweep points (the analysis returns the exact count) */
int qp_sweep_maxpts(int stepType, int np, double fstart, double fstop)
{
    if (stepType == 1) return (int)(np * log10(fstop/fstart)) + 3;        /* dec */
    if (stepType == 2) return (int)(np * log(fstop/fstart)/log(2.0)) + 3; /* oct */
    return np + 1;                                                        /* lin */
}

/* build an ngspice plot named `plotname` with a frequency scale + nvec real data
 * vectors (data is point-major: data[p*nvec + k]). Shared by qpac/qpnoise/qpxf sweeps. */
void qp_emit_plot(const char *plotname, const char *title, double *freqs, int npts,
                  char **vnames, int nvec, double *data)
{
    struct plot *pl = plot_alloc((char *) plotname);
    struct dvec *sc;
    int p, k;
    pl->pl_name = copy((char *) title);
    pl->pl_title = copy((char *) title);
    plot_new(pl);
    plot_setcur(pl->pl_typename);
    sc = dvec_alloc(copy("frequency"), SV_FREQUENCY, (short)(VF_REAL | VF_PERMANENT), npts, NULL);
    for (p = 0; p < npts; p++) sc->v_realdata[p] = freqs[p];
    vec_new(sc);                                     /* first permanent vector -> scale */
    for (k = 0; k < nvec; k++) {
        struct dvec *v = dvec_alloc(copy(vnames[k]), SV_NOTYPE, (short)(VF_REAL | VF_PERMANENT), npts, NULL);
        for (p = 0; p < npts; p++) v->v_realdata[p] = data[(size_t)p*(size_t)nvec + (size_t)k];
        vec_new(v);
    }
}

void
com_qpac(wordlist *wl)
{
    CKTcircuit *ckt;
    double f_in;
    int    verbose, err;

    if (!ft_curckt || !ft_curckt->ci_ckt) {
        fprintf(cp_err, "Error: qpac: there is no circuit loaded.\n");
        return;
    }
    ckt = ft_curckt->ci_ckt;

    if (!wl || !wl->wl_word) {
        fprintf(cp_err, "Usage: qpac <f_in> | qpac <dec|oct|lin> <N> <fstart> <fstop>   (run `qpss ... hb` first)\n");
        return;
    }
    {   /* sweep form: qpac <dec|oct|lin> <N> <fstart> <fstop> -> plot vs f_in */
        int st = qp_steptype(wl->wl_word);
        if (st >= 0) {
            double fstart, fstop, *freqs, *data; int np, npts, maxpts, numNames = 0, i;
            IFuid *nameList = NULL; char **vn;
            if (!wl->wl_next || !wl->wl_next->wl_next || !wl->wl_next->wl_next->wl_next) {
                fprintf(cp_err, "Usage: qpac <dec|oct|lin> <N> <fstart> <fstop>\n"); return;
            }
            np = (int) qpacnum(wl->wl_next->wl_word);
            fstart = qpacnum(wl->wl_next->wl_next->wl_word);
            fstop  = qpacnum(wl->wl_next->wl_next->wl_next->wl_word);
            if (np < 1 || fstart <= 0.0 || fstop < fstart) { fprintf(cp_err, "Error: qpac: bad sweep spec.\n"); return; }
            if (CKTnames(ckt, &numNames, &nameList) != OK || numNames < 1) { fprintf(cp_err, "Error: qpac: no nodes.\n"); return; }
            maxpts = qp_sweep_maxpts(st, np, fstart, fstop);
            freqs = TMALLOC(double, maxpts);
            data  = TMALLOC(double, (size_t)maxpts * (size_t)numNames);
            npts = QPACsweep(ckt, st, np, fstart, fstop, freqs, data);
            if (npts > 0) {
                vn = TMALLOC(char *, numNames);
                for (i = 0; i < numNames; i++) vn[i] = (char *) nameList[i];
                qp_emit_plot("qpac", "QPAC Analysis", freqs, npts, vn, numNames, data);
                fprintf(cp_out, "qpac: swept %d points into a new plot (now current); `plot <node>` to view the per-node response vs f_in.\n", npts);
                FREE(vn);
            } else fprintf(cp_err, "qpac: sweep did not complete.\n");
            tfree(nameList); FREE(freqs); FREE(data);
            return;
        }
    }
    f_in = qpacnum(wl->wl_word);
    if (f_in <= 0.0) {
        fprintf(cp_err, "Error: qpac: need f_in > 0.\n");
        return;
    }

    verbose = cp_getvar("qpac_verbose", CP_BOOL, NULL, 0);
    err = QPACanalyze(ckt, f_in, verbose ? 1 : 0);
    if (err != OK)
        fprintf(cp_err, "qpac: quasi-periodic AC did not complete (error %d).\n", err);
}
