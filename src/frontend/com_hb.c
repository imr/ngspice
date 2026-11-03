/**********
Enhancement-134: Harmonic Balance -- `hb <f0> <K> [points] [maxiter]`.

Single-tone harmonic balance: find the periodic steady state in the FREQUENCY
domain by Newton, instead of integrating in time. Each node voltage is a truncated
Fourier series V(t)=sum_{k=-K..K} V_k e^{jk w0 t}; the KCL residual at each
node/harmonic is driven to zero with the E-121 conversion matrix as the Jacobian.
The heavy lifting is in HBanalyze() (spicelib/analysis/dcpss.c, which reuses the
conversion matrix + dense complex solver); this command parses the arguments,
makes sure the circuit is built, and runs it.
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cpdefs.h"
#include "ngspice/cktdefs.h"
#include "ngspice/ftedefs.h"
#include "ngspice/fteext.h"
#include "ngspice/wordlist.h"
#include "ngspice/cpextern.h"
#include "ngspice/dvec.h"        /* Enhancement-209: dvec_alloc + VF_/SV_ for result vectors */
#include "ngspice/sim.h"

#include "circuits.h"
#include "com_hb.h"

static double hbnum(const char *w)
{
    char *s = (char *) w;
    double v = 0.0;
    if (ft_numparse(&s, FALSE, &v) < 0)
        v = atof(w);
    return v;
}

/* Enhancement-209: publish the harmonic-balance spectrum as nutmeg vectors so the
   user can plot / print / wrdata it directly. A fresh "hb" plot holds a real scale
   `hbfrequency` (0, f0, 2f0, ..., K f0) and one COMPLEX vector per node, carrying
   the single-sided amplitude (|.| and phase match the printed table). The plot is
   left current, so `plot mag(out)` / `print out` / `wrdata sp v(out)` work at once. */
static void
hb_publish(CKTcircuit *ckt, const struct hbspectrum *sp)
{
    int numNames = 0, error, j, k, nv = 0;
    int N = sp->N, K = sp->K;
    IFuid *nameList = NULL;
    struct plot *pl;
    struct dvec *fv;

    if (!sp->Vr || !sp->Vi || N <= 0 || K < 1)
        return;
    error = CKTnames(ckt, &numNames, &nameList);
    if (error || numNames <= 0) {
        if (nameList)
            tfree(nameList);
        return;
    }

    pl = plot_alloc("hb");
    pl->pl_name  = copy("Harmonic Balance");
    pl->pl_title = copy((ft_curckt && ft_curckt->ci_name) ? ft_curckt->ci_name : "hb");
    plot_new(pl);
    plot_setcur(pl->pl_typename);

    /* the harmonic-frequency scale, created first so it becomes the plot scale */
    fv = dvec_alloc(copy("hbfrequency"), SV_FREQUENCY,
                    (short) (VF_REAL | VF_PERMANENT), K + 1, NULL);
    for (k = 0; k <= K; k++)
        fv->v_realdata[k] = k * sp->f0;
    vec_new(fv);

    for (j = 0; j < numNames && j < N; j++) {
        const char *nm = (const char *) nameList[j];
        int isI = (nm && strstr(nm, "#branch") != NULL);
        struct dvec *v = dvec_alloc(copy(nm), isI ? SV_CURRENT : SV_VOLTAGE,
                                    (short) (VF_COMPLEX | VF_PERMANENT), K + 1, NULL);
        for (k = 0; k <= K; k++) {
            double sc = (k == 0) ? 1.0 : 2.0;   /* single-sided amplitude */
            size_t idx = (size_t) (k + K) * (size_t) N + (size_t) j;
            v->v_compdata[k].cx_real = sc * sp->Vr[idx];
            v->v_compdata[k].cx_imag = sc * sp->Vi[idx];
        }
        vec_new(v);
        nv++;
    }
    tfree(nameList);

    fprintf(cp_out, "hb: spectrum stored in the current 'hb' plot -- 'hbfrequency' + "
                    "%d node vector%s (try  plot mag(<node>)  or  wrdata out <node>).\n",
            nv, nv == 1 ? "" : "s");
}

void
com_hb(wordlist *wl)
{
    CKTcircuit *ckt;
    double f0, tol = 1e-10;
    int    K, P = 0, maxiter = 50, verbose, err;

    if (!ft_curckt || !ft_curckt->ci_ckt) {
        fprintf(cp_err, "Error: hb: there is no circuit loaded.\n");
        return;
    }
    ckt = ft_curckt->ci_ckt;

    if (!wl || !wl->wl_next) {
        fprintf(cp_err, "Usage: hb <f0> <K> [points] [maxiter]\n");
        return;
    }
    f0 = hbnum(wl->wl_word);
    K  = (int) hbnum(wl->wl_next->wl_word);
    if (wl->wl_next->wl_next) {
        P = (int) hbnum(wl->wl_next->wl_next->wl_word);
        if (wl->wl_next->wl_next->wl_next)
            maxiter = (int) hbnum(wl->wl_next->wl_next->wl_next->wl_word);
    }
    if (f0 <= 0.0 || K < 1) {
        fprintf(cp_err, "Error: hb: need f0 > 0 and K >= 1.\n");
        return;
    }

    /* Honor `.option klu`: a bare `hb` builds the circuit here rather than via
     * cktdojob (which is where a normal analysis copies task->TSKkluMODE), so pull
     * the KLU mode from the task before CKTsetup wires the matrix -- else the matrix
     * defaults to Sparse and the `.option klu` request is silently ignored. */
#ifdef KLU
    if (ft_curckt->ci_defTask &&
        (ckt->CKTmatrix == NULL || SMPmatSize(ckt->CKTmatrix) <= 0))
        ckt->CKTkluMODE = ft_curckt->ci_defTask->TSKkluMODE;
#endif

    /* make sure the circuit is built (matrix + states allocated) */
    if (ckt->CKTmatrix == NULL || SMPmatSize(ckt->CKTmatrix) <= 0) {
        if ((err = CKTsetup(ckt)) != OK || (err = CKTtemp(ckt)) != OK) {
            fprintf(cp_err, "Error: hb: circuit setup failed.\n");
            return;
        }
    }

    verbose = cp_getvar("hb_verbose", CP_BOOL, NULL, 0);
    ft_curckt->ci_curTask = ft_curckt->ci_defTask;
    ckt->CKTcurJob = ft_curckt->ci_defTask ? ft_curckt->ci_defTask->jobs : NULL;

    {
        struct hbspectrum sp;
        memset(&sp, 0, sizeof sp);
        err = HBanalyze(ckt, f0, K, P, maxiter, tol, verbose ? 1 : 0, &sp);
        if (err != OK)
            fprintf(cp_err, "hb: harmonic balance did not complete (error %d).\n", err);
        else
            hb_publish(ckt, &sp);
        FREE(sp.Vr);
        FREE(sp.Vi);
    }
}
