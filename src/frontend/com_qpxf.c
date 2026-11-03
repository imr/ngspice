/**********
Enhancement-141: two-tone small-signal QPXF (quasi-periodic transfer function) --
`qpxf <output_node> <f_in>`.

The adjoint of QPAC: around the QPSS operating point retained by a prior
`qpss <expr> <f1> <f2> hb`, one adjoint solve of the 2-D conversion matrix gives the
transfer from an input at every sideband f_in + k1*f1 + k2*f2 to the chosen output at
f_in. By reciprocity the sideband-(0,0) transfer equals the QPAC response at that node.
The engine is QPXFanalyze() (spicelib/analysis/dcpss.c); this command resolves the
output node and runs it.
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cpdefs.h"
#include "ngspice/cktdefs.h"
#include "ngspice/ftedefs.h"
#include "ngspice/fteext.h"
#include "ngspice/wordlist.h"
#include "ngspice/cpextern.h"

#include "com_qpxf.h"

extern int qp_steptype(const char* w);
extern int qp_sweep_maxpts(int stepType, int np, double fstart, double fstop);
extern void qp_emit_plot(const char* plotname, const char* title, double* freqs, int npts,
    char** vnames, int nvec, double* data);

static double qpxfnum(const char *w)
{
    char *s = (char *) w;
    double v = 0.0;
    if (ft_numparse(&s, FALSE, &v) < 0)
        v = atof(w);
    return v;
}

static int qpxf_node(CKTcircuit *ckt, const char *name)
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
com_qpxf(wordlist *wl)
{
    CKTcircuit *ckt;
    double f_in;
    int    outNode, verbose, err;

    if (!ft_curckt || !ft_curckt->ci_ckt) {
        fprintf(cp_err, "Error: qpxf: there is no circuit loaded.\n");
        return;
    }
    ckt = ft_curckt->ci_ckt;
    if (!wl || !wl->wl_next) {
        fprintf(cp_err, "Usage: qpxf <output_node> <f_in>   (run `qpss <expr> <f1> <f2> hb` first)\n");
        return;
    }
    outNode = qpxf_node(ckt, wl->wl_word);
    if (outNode <= 0) { fprintf(cp_err, "Error: qpxf: unknown output node '%s'.\n", wl->wl_word); return; }
    {   /* sweep form: qpxf <out> <dec|oct|lin> <N> <fstart> <fstop> -> xf/xf_conv plot */
        int st = qp_steptype(wl->wl_next->wl_word);
        if (st >= 0) {
            double fstart, fstop, *freqs, *data; int np, npts, maxpts;
            char *vn[2] = { (char*)"xf", (char*)"xf_conv" };
            wordlist *w = wl->wl_next->wl_next;
            if (!w || !w->wl_next || !w->wl_next->wl_next) {
                fprintf(cp_err, "Usage: qpxf <output_node> <dec|oct|lin> <N> <fstart> <fstop>\n"); return;
            }
            np = (int) qpxfnum(w->wl_word);
            fstart = qpxfnum(w->wl_next->wl_word);
            fstop  = qpxfnum(w->wl_next->wl_next->wl_word);
            if (np < 1 || fstart <= 0.0 || fstop < fstart) { fprintf(cp_err, "Error: qpxf: bad sweep spec.\n"); return; }
            maxpts = qp_sweep_maxpts(st, np, fstart, fstop);
            freqs = TMALLOC(double, maxpts); data = TMALLOC(double, (size_t)maxpts*2);
            npts = QPXFsweep(ckt, outNode, st, np, fstart, fstop, freqs, data);
            if (npts > 0) {
                qp_emit_plot("qpxf", "QPXF Analysis", freqs, npts, vn, 2, data);
                fprintf(cp_out, "qpxf: swept %d points into a new plot (now current); `plot xf xf_conv` to view.\n", npts);
            } else fprintf(cp_err, "qpxf: sweep did not complete.\n");
            FREE(freqs); FREE(data);
            return;
        }
    }
    f_in = qpxfnum(wl->wl_next->wl_word);
    if (f_in <= 0.0) { fprintf(cp_err, "Error: qpxf: need f_in > 0.\n"); return; }

    verbose = cp_getvar("qpxf_verbose", CP_BOOL, NULL, 0);
    err = QPXFanalyze(ckt, outNode, f_in, verbose ? 1 : 0);
    if (err != OK)
        fprintf(cp_err, "qpxf: quasi-periodic transfer function did not complete (error %d).\n", err);
}
