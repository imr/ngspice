/**********
 Author: 2010-05 Stefano Perticaroli ``spertica''
 First Review: 2012-02 Francesco Lannutti and Stefano Perticaroli ``spertica''
 Second Review: 2012-10 Stefano Perticaroli ``spertica'' and Francesco Lannutti
**********/

/* Include files for the PSS analysis */
#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "cktaccept.h"
#include "ngspice/pssdefs.h"
#include "ngspice/devdefs.h"   /* Enhancement-120: DEVbindCSCComplex for the KLU AC stamp */
#include "ngspice/smpdefs.h"   /* Enhancement-120: SMPfindElt to read the Jacobian */
#include "ngspice/spmatrix.h"  /* Enhancement-120: spSetComplex (Sparse complex mode) */
#include "ngspice/noisedef.h"  /* Enhancement-124: NOISEAN/Ndata + CKTnoise for pnoise */
#include "ngspice/sperror.h"
#include "ngspice/fteext.h"
#include "ngspice/cpextern.h"   /* Enhancement-193: cp_getvar for the `sqrnoise` flag */
#ifdef RFSPICE
#include "vsrc/vsrcdefs.h"             /* Enhancement-132: RF port fields (z0, ki, branch) */
#include "isrc/isrcdefs.h"             /* Enhancement-176: driven-mode source detection */
#include "../maths/dense/dense.h"      /* Enhancement-132: complex S = B*A^-1 */
#include "../maths/dense/denseinlines.h"
#endif

#ifdef XSPICE
/* gtri - add - wbk - Add headers */
#include "ngspice/miftypes.h"

#include "ngspice/evt.h"
#include "ngspice/enh.h"
#include "ngspice/mif.h"
#include "ngspice/evtproto.h"
#include "ngspice/ipctiein.h"
/* gtri - end - wbk - Add headers */
#endif

extern char* eng(double value, int digits, bool numeric, bool bytes);

#define INIT_STATS() \
do { \
    startTime = SPfrontEnd->IFseconds();        \
    startIters = ckt->CKTstat->STATnumIter;     \
    startdTime = ckt->CKTstat->STATdecompTime;  \
    startsTime = ckt->CKTstat->STATsolveTime;   \
    startlTime = ckt->CKTstat->STATloadTime;    \
    startkTime = ckt->CKTstat->STATsyncTime;    \
} while(0)

#define UPDATE_STATS(analysis) \
do { \
    ckt->CKTcurrentAnalysis = analysis; \
    ckt->CKTstat->STATtranTime += SPfrontEnd->IFseconds() - startTime; \
    ckt->CKTstat->STATtranIter += ckt->CKTstat->STATnumIter - startIters; \
    ckt->CKTstat->STATtranDecompTime += ckt->CKTstat->STATdecompTime - startdTime; \
    ckt->CKTstat->STATtranSolveTime += ckt->CKTstat->STATsolveTime - startsTime; \
    ckt->CKTstat->STATtranLoadTime += ckt->CKTstat->STATloadTime - startlTime; \
    ckt->CKTstat->STATtranSyncTime += ckt->CKTstat->STATsyncTime - startkTime; \
} while(0)

/* Enhancement-117: gate the shooting-loop trace behind `set ngdebug`.
 * The PSS shooting method prints per-iteration diagnostics (frequency estimate,
 * residual, breakpoint bookkeeping, delta control) that are invaluable while
 * debugging the method but overwhelm normal use -- a single .pss run emitted
 * ~230 lines of trace. Routed through PSSDBG they stay available (ngdebug on)
 * without polluting production output; genuine results/errors remain plain
 * fprintf. */
#define PSSDBG(...) do { if (ft_ngdebug) fprintf(stderr, __VA_ARGS__); } while(0)


/* Define some useful macro */
#define HISTORY 1024
#define GF_LAST 313

//#define PSSDEBUG
//#define STEPDEBUG

static int
DFT(long int, int, double *, double *, double *, double, double *, double *, double *, double *, double *);


/* Enhancement-120: periodic small-signal Jacobian harmonics.
 *
 * PAC/pnoise/PXF linearize around the periodic operating point and solve a
 * harmonic conversion matrix whose blocks are the harmonics G_k, C_k of the
 * periodically time-varying device Jacobian G(t) = dI/dV, C(t) = dQ/dV. This
 * routine builds the first piece: it walks the retained operating point (E-119),
 * and at each stored sample restores that instant's node voltages + device
 * states, recomputes the small-signal linearization (CKTload with MODEINITSMSIG),
 * and stamps G + jC into the complex matrix (CKTacLoad at omega = 1). The
 * (osc,osc) diagonal read back is the osc node's conductance g(t) (real part) and
 * capacitance c(t) (imag part); their DFT gives the periodic Jacobian's harmonics
 * -- flat (DC only) for a linear circuit, rich for a pumped nonlinear one. The
 * osc-node diagonal is reported as a verifiable slice of the full G_k/C_k the PAC
 * conversion matrix (E-121) will be assembled from. */
static void
pss_jacobian_report(CKTcircuit *ckt, PSSan *job)
{
    long P = job->PSSopPoints, s;
    int  msize = job->PSSopMsize, ns = job->PSSopNumStates;
    int  onode = job->PSSoscNode ? job->PSSoscNode->number : 0;
    int  i, K = ckt->CKTharms;
    double thd;
    double *gt, *ct, *tt, *frq, *mag, *phs, *nmag, *nphs;

    if (onode <= 0 || onode > msize || P <= 0 || K <= 0)
        return;

    gt  = TMALLOC(double, P);   ct   = TMALLOC(double, P);   tt   = TMALLOC(double, P);
    frq = TMALLOC(double, K);   mag  = TMALLOC(double, K);   phs  = TMALLOC(double, K);
    nmag= TMALLOC(double, K);   nphs = TMALLOC(double, K);

    /* the complex AC stamps need the matrix in complex mode, otherwise
     * SMPcClear (spClear) leaves the imaginary part uncleared and C(t)
     * accumulates across samples. */
#ifdef KLU
    if (ckt->CKTmatrix->CKTkluMODE) {
        if (!ckt->CKTmatrix->SMPkluMatrix->KLUmatrixIsComplex) {
            for (i = 0; i < DEVmaxnum; i++)
                if (DEVices[i] && DEVices[i]->DEVbindCSCComplex && ckt->CKThead[i])
                    DEVices[i]->DEVbindCSCComplex(ckt->CKThead[i], ckt);
            ckt->CKTmatrix->SMPkluMatrix->KLUmatrixIsComplex = KLUMatrixComplex;
        }
    } else
#endif
        spSetComplex(ckt->CKTmatrix->SPmatrix);

    for (s = 0; s < P; s++) {
        double *e;
        for (i = 1; i <= msize; i++)
            ckt->CKTrhsOld[i] = job->PSSopVoltages[(i - 1) + s * msize];
        ckt->CKTrhsOld[0] = 0.0;
        if (ns > 0)
            memcpy(ckt->CKTstate0, job->PSSopStates + (size_t)s * (size_t)ns,
                   (size_t)ns * sizeof(double));

        /* recompute the device linearization at this instant's bias */
        ckt->CKTmode = (ckt->CKTmode & MODEUIC) | MODEDCOP | MODEINITSMSIG;
        CKTload(ckt);

        /* stamp G + jC (omega = 1 so the imaginary part is exactly C) */
        ckt->CKTomega = 1.0;
        ckt->CKTmode = (ckt->CKTmode & MODEUIC) | MODEAC;
        CKTacLoad(ckt);

        e = (double *) SMPfindElt(ckt->CKTmatrix, onode, onode, 0);
        gt[s] = e ? e[0] : 0.0;   /* Real = conductance  G(t) */
        ct[s] = e ? e[1] : 0.0;   /* Imag = capacitance  C(t) (omega = 1) */
        tt[s] = job->PSSopTimes[s];
    }

    fprintf(stderr, "periodic small-signal Jacobian at osc node (%ld samples, %d harmonics):\n", P, K);
    DFT(P, K, &thd, tt, gt, job->PSSopFreq, frq, mag, phs, nmag, nphs);
    fprintf(stderr, "  G(t): DC = %.6g S", mag[0]);
    for (i = 1; i < K && i < 4; i++) fprintf(stderr, ", |G%d| = %.4g", i, mag[i]);
    fprintf(stderr, "\n");
    DFT(P, K, &thd, tt, ct, job->PSSopFreq, frq, mag, phs, nmag, nphs);
    fprintf(stderr, "  C(t): DC = %.6g F", mag[0]);
    for (i = 1; i < K && i < 4; i++) fprintf(stderr, ", |C%d| = %.4g", i, mag[i]);
    fprintf(stderr, "\n");

    FREE(gt); FREE(ct); FREE(tt);
    FREE(frq); FREE(mag); FREE(phs); FREE(nmag); FREE(nphs);
}


/* Enhancement-121: dense complex linear solve A x = b (Gaussian elimination with
 * partial pivoting), used for the small harmonic conversion matrix below. A is
 * n x n row-major in split real/imag arrays (Ar, Ai); b is length n in (br, bi)
 * and is overwritten with the solution x. Returns 0 on success, 1 if singular.
 * The conversion matrix is (2K+1)*msize -- tiny for the circuits PAC targets, so
 * a direct dense factor is simplest and exact; a production PAC on large circuits
 * would assemble this as a sparse block system instead. */
static int
pss_csolve(int n, double *Ar, double *Ai, double *br, double *bi)
{
    int i, j, k, piv;

    for (k = 0; k < n; k++) {
        /* partial pivot: largest |A[i][k]| in the remaining column */
        double amax = -1.0;
        piv = k;
        for (i = k; i < n; i++) {
            double m = Ar[i*n+k]*Ar[i*n+k] + Ai[i*n+k]*Ai[i*n+k];
            if (m > amax) { amax = m; piv = i; }
        }
        if (amax <= 0.0)
            return 1;                       /* singular */
        if (piv != k) {                     /* swap rows piv <-> k */
            for (j = 0; j < n; j++) {
                double t;
                t = Ar[piv*n+j]; Ar[piv*n+j] = Ar[k*n+j]; Ar[k*n+j] = t;
                t = Ai[piv*n+j]; Ai[piv*n+j] = Ai[k*n+j]; Ai[k*n+j] = t;
            }
            { double t; t = br[piv]; br[piv] = br[k]; br[k] = t;
                       t = bi[piv]; bi[piv] = bi[k]; bi[k] = t; }
        }
        /* eliminate below the pivot */
        {
            double dr = Ar[k*n+k], di = Ai[k*n+k], den = dr*dr + di*di;
            for (i = k+1; i < n; i++) {
                double fr = (Ar[i*n+k]*dr + Ai[i*n+k]*di) / den;   /* A[i][k]/A[k][k] */
                double fi = (Ai[i*n+k]*dr - Ar[i*n+k]*di) / den;
                for (j = k; j < n; j++) {
                    double pr = fr*Ar[k*n+j] - fi*Ai[k*n+j];
                    double pi = fr*Ai[k*n+j] + fi*Ar[k*n+j];
                    Ar[i*n+j] -= pr; Ai[i*n+j] -= pi;
                }
                { double pr = fr*br[k] - fi*bi[k];
                  double pi = fr*bi[k] + fi*br[k];
                  br[i] -= pr; bi[i] -= pi; }
            }
        }
    }
    /* back-substitution */
    for (k = n-1; k >= 0; k--) {
        double sr = br[k], si = bi[k];
        double dr, di, den;
        for (j = k+1; j < n; j++) {
            sr -= Ar[k*n+j]*br[j] - Ai[k*n+j]*bi[j];
            si -= Ar[k*n+j]*bi[j] + Ai[k*n+j]*br[j];
        }
        dr = Ar[k*n+k]; di = Ai[k*n+k]; den = dr*dr + di*di;
        br[k] = (sr*dr + si*di) / den;
        bi[k] = (si*dr - sr*di) / den;
    }
    return 0;
}


/* Enhancement-121: periodic AC (PAC) conversion-matrix engine.
 *
 * A circuit linearized about its PSS steady state has a T-periodic Jacobian, so a
 * small tone at f_in produces responses not only at f_in but at every sideband
 * f_in + k*f0 (k = -M..M). Collecting the harmonics G_k, C_k of the periodic
 * Jacobian (E-120, now for every matrix entry, not just the osc diagonal) into a
 * block matrix gives the harmonic conversion matrix H, block (n,m):
 *
 *     H_{nm} = G_{n-m} + j*omega_m*C_{n-m},   omega_m = 2*pi*(f_in + m*f0)
 *
 * of size (2M+1)*N. Solving H X = B for a stimulus B injected at one sideband
 * yields the responses X at all sidebands -- the conversion gains. This routine
 * assembles H, injects a unit current at the osc node in the 0-th sideband, solves,
 * and reports the response magnitude at the -1/0/+1 sidebands. For a *linear*
 * circuit the off-diagonal harmonics G_k,C_k (k!=0) vanish, so H is block-diagonal,
 * the 0-block is exactly the ordinary AC matrix at f_in, and the result is the AC
 * driving-point response at f_in with zero conversion to the other sidebands --
 * the verifiable slice. A pumped nonlinear circuit fills the off-diagonal blocks
 * and mixes energy between sidebands (real conversion gain). */
/* Enhancement-122: the periodic Jacobian harmonics G_k, C_k, extracted once from
 * the retained operating point and shared by the single-frequency diagnostic
 * (E-121, .pss) and the frequency sweep (E-122, .pac). */
struct pac_harm {
    int N;                          /* matrix size (unknowns) */
    int M;                          /* sidebands each side; harmonics span -2M..2M */
    int H;                          /* max harmonic index = 2M */
    int nnz;                        /* structural nonzeros of the Jacobian */
    int Ntot;                       /* (2M+1)*N -- conversion-matrix dimension */
    int *rr, *cc;                   /* nonzero row/col (1-based) */
    double *Gmr, *Gmi, *Cmr, *Cmi;  /* [nnz*(H+1)] complex harmonics G_h, C_h */
    /* Enhancement-123: the small-signal source RHS captured from CKTacLoad (the AC
     * stamp of netlist `AC`-flagged sources), used as the sideband-0 stimulus B_0
     * when present -- the source-referenced PAC input, else a unit current at the
     * osc node is injected as a fallback. Bias-independent, so captured once. */
    double *B0r, *B0i;              /* [N] source AC RHS (0-based, node j -> row j+1) */
    int    has_src;                 /* 1 if any netlist source stamped an AC value */
};

/* Walk the retained operating point, sample every Jacobian nonzero's G(t), C(t)
 * over the period and complex-DFT them to harmonics G_h, C_h (h = 0..2M). On
 * success fills hd (which then owns its arrays) and returns 0; else returns 1. */
static int
pac_extract_harmonics(CKTcircuit *ckt, PSSan *job, int M, struct pac_harm *hd)
{
    long   P = job->PSSopPoints, s;
    int    N = job->PSSopMsize, ns = job->PSSopNumStates;
    int    H = 2 * M, i, r, c, e, h, nnz;
    int    *rr, *cc;
    double *Gt, *Ct, *cw, *sw, *Gmr, *Gmi, *Cmr, *Cmi, *B0r, *B0i;
    int    has_src = 0;

    memset(hd, 0, sizeof(*hd));
    if (P <= 0 || N <= 0 || M < 1)
        return 1;

    /* matrix must be in complex mode so CKTacLoad's SMPcClear clears the imag part
     * (else C(t) accumulates across samples -- see E-120). */
#ifdef KLU
    if (ckt->CKTmatrix->CKTkluMODE) {
        if (!ckt->CKTmatrix->SMPkluMatrix->KLUmatrixIsComplex) {
            for (i = 0; i < DEVmaxnum; i++)
                if (DEVices[i] && DEVices[i]->DEVbindCSCComplex && ckt->CKThead[i])
                    DEVices[i]->DEVbindCSCComplex(ckt->CKThead[i], ckt);
            ckt->CKTmatrix->SMPkluMatrix->KLUmatrixIsComplex = KLUMatrixComplex;
        }
    } else
#endif
        spSetComplex(ckt->CKTmatrix->SPmatrix);

    /* establish the matrix structure: stamp G + jC at sample 0's bias */
    for (i = 1; i <= N; i++)
        ckt->CKTrhsOld[i] = job->PSSopVoltages[(i - 1)];
    ckt->CKTrhsOld[0] = 0.0;
    if (ns > 0)
        memcpy(ckt->CKTstate0, job->PSSopStates, (size_t)ns * sizeof(double));
    ckt->CKTmode = (ckt->CKTmode & MODEUIC) | MODEDCOP | MODEINITSMSIG;
    CKTload(ckt);
    ckt->CKTomega = 1.0;
    ckt->CKTmode = (ckt->CKTmode & MODEUIC) | MODEAC;
    CKTacLoad(ckt);

    /* Enhancement-123: capture the small-signal source RHS. CKTacLoad clears
     * CKTrhs/CKTirhs then lets each device stamp; a netlist source with an `AC`
     * spec stamps its (bias-independent) AC value here -- that vector is the
     * source-referenced PAC stimulus B_0. */
    B0r = TMALLOC(double, N);
    B0i = TMALLOC(double, N);
    for (i = 1; i <= N; i++) {
        B0r[i - 1] = ckt->CKTrhs[i];
        B0i[i - 1] = ckt->CKTirhs[i];
        if (B0r[i - 1] != 0.0 || B0i[i - 1] != 0.0)
            has_src = 1;
    }

    /* enumerate the structural nonzeros (SMPfindElt does not create) */
    nnz = 0;
    for (r = 1; r <= N; r++)
        for (c = 1; c <= N; c++)
            if (SMPfindElt(ckt->CKTmatrix, r, c, 0))
                nnz++;
    if (nnz <= 0)
        return 1;
    rr = TMALLOC(int, nnz);
    cc = TMALLOC(int, nnz);
    e = 0;
    for (r = 1; r <= N; r++)
        for (c = 1; c <= N; c++)
            if (SMPfindElt(ckt->CKTmatrix, r, c, 0)) { rr[e] = r; cc[e] = c; e++; }

    /* sample every nonzero of the periodic Jacobian G(t) + jC(t) over one period */
    Gt = TMALLOC(double, (size_t)nnz * (size_t)P);
    Ct = TMALLOC(double, (size_t)nnz * (size_t)P);
    for (s = 0; s < P; s++) {
        for (i = 1; i <= N; i++)
            ckt->CKTrhsOld[i] = job->PSSopVoltages[(i - 1) + s * N];
        ckt->CKTrhsOld[0] = 0.0;
        if (ns > 0)
            memcpy(ckt->CKTstate0, job->PSSopStates + (size_t)s * (size_t)ns,
                   (size_t)ns * sizeof(double));
        ckt->CKTmode = (ckt->CKTmode & MODEUIC) | MODEDCOP | MODEINITSMSIG;
        CKTload(ckt);
        ckt->CKTomega = 1.0;
        ckt->CKTmode = (ckt->CKTmode & MODEUIC) | MODEAC;
        CKTacLoad(ckt);
        for (e = 0; e < nnz; e++) {
            double *el = (double *) SMPfindElt(ckt->CKTmatrix, rr[e], cc[e], 0);
            Gt[(size_t)e * (size_t)P + (size_t)s] = el ? el[0] : 0.0;
            Ct[(size_t)e * (size_t)P + (size_t)s] = el ? el[1] : 0.0;
        }
    }

    /* complex DFT of each entry: harmonics h = 0..H (G_{-h} = conj(G_h) for real
     * G(t)). Uniform sampling over the period, so index-based twiddles suffice. */
    cw = TMALLOC(double, (size_t)(H + 1) * (size_t)P);
    sw = TMALLOC(double, (size_t)(H + 1) * (size_t)P);
    for (h = 0; h <= H; h++)
        for (s = 0; s < P; s++) {
            double ang = 2.0 * M_PI * (double)h * (double)s / (double)P;
            cw[(size_t)h * (size_t)P + (size_t)s] = cos(ang);
            sw[(size_t)h * (size_t)P + (size_t)s] = sin(ang);
        }
    Gmr = TMALLOC(double, (size_t)nnz * (size_t)(H + 1));
    Gmi = TMALLOC(double, (size_t)nnz * (size_t)(H + 1));
    Cmr = TMALLOC(double, (size_t)nnz * (size_t)(H + 1));
    Cmi = TMALLOC(double, (size_t)nnz * (size_t)(H + 1));
    for (e = 0; e < nnz; e++)
        for (h = 0; h <= H; h++) {
            double gr = 0, gi = 0, cr = 0, ci = 0;
            for (s = 0; s < P; s++) {
                double cs = cw[(size_t)h * (size_t)P + (size_t)s];
                double sn = sw[(size_t)h * (size_t)P + (size_t)s];
                double gv = Gt[(size_t)e * (size_t)P + (size_t)s];
                double cv = Ct[(size_t)e * (size_t)P + (size_t)s];
                gr += gv * cs;  gi -= gv * sn;
                cr += cv * cs;  ci -= cv * sn;
            }
            Gmr[(size_t)e * (size_t)(H + 1) + (size_t)h] = gr / (double)P;
            Gmi[(size_t)e * (size_t)(H + 1) + (size_t)h] = gi / (double)P;
            Cmr[(size_t)e * (size_t)(H + 1) + (size_t)h] = cr / (double)P;
            Cmi[(size_t)e * (size_t)(H + 1) + (size_t)h] = ci / (double)P;
        }

    FREE(Gt); FREE(Ct); FREE(cw); FREE(sw);

    hd->N = N;  hd->M = M;  hd->H = H;  hd->nnz = nnz;  hd->Ntot = (2*M + 1) * N;
    hd->rr = rr;  hd->cc = cc;
    hd->Gmr = Gmr;  hd->Gmi = Gmi;  hd->Cmr = Cmr;  hd->Cmi = Cmi;
    hd->B0r = B0r;  hd->B0i = B0i;  hd->has_src = has_src;
    return 0;
}

static void
pac_free_harmonics(struct pac_harm *hd)
{
    FREE(hd->rr);  FREE(hd->cc);
    FREE(hd->Gmr); FREE(hd->Gmi); FREE(hd->Cmr); FREE(hd->Cmi);
    FREE(hd->B0r); FREE(hd->B0i);
}

/* Enhancement-121: assemble the dense (2M+1)N complex conversion matrix at input
 * frequency f_in into (Ar,Ai). Shared by the PAC (pac_solve_at) and PSP
 * (psp_solve_port) small-signal solves and by the HB/HBOSC Newton (residual and
 * Jacobian).
 *
 * The reactive frequency multiplier differs between the two uses (RF-audit fix):
 *
 *   smallsig=1 (LPTV small-signal: PAC/PSP/pnoise/pxf):
 *       H_{nm} = G_{n-m} + j*omega_n*C_{n-m}      (ROW = output sideband)
 *     The small signal rides a FROZEN periodic C(t) = dQ/dv|orbit, so
 *     di = d/dt[C(t) dv] = Cdot*dv + C*dv_dot; by the product rule the n-th
 *     sideband is j*[(n-m)w0 + w_m]*C_{n-m}*X_m = j*w_n*C_{n-m}*X_m. Using the
 *     column frequency here silently DROPS the parametric-pumping term Cdot*dv
 *     (a pumped varactor would not convert).
 *
 *   smallsig=0 (harmonic-balance residual, f_in = 0):
 *       H_{nm} = G_{n-m} + j*omega_m*C_{n-m}      (COLUMN = chain rule)
 *     The residual's reactive current is the exact d/dt Q(v(t)) = C(v(t))*vdot,
 *     whose n-th harmonic is sum_m C_{n-m} * (j*w_m*V_m) -- here the COLUMN
 *     frequency is the correct one (and the same form serves as the standard
 *     frozen-C quasi-Newton Jacobian). */
static void
pac_build_matrix(struct pac_harm *hd, double f0, double f_in, double *Ar, double *Ai,
                 int smallsig)
{
    int    N = hd->N, M = hd->M, H = hd->H, nnz = hd->nnz, Ntot = hd->Ntot;
    int    ni, mi, n, mm, e;

    memset(Ar, 0, (size_t)Ntot * (size_t)Ntot * sizeof(double));
    memset(Ai, 0, (size_t)Ntot * (size_t)Ntot * sizeof(double));
    for (ni = 0; ni <= 2*M; ni++) {
        n = ni - M;
        for (mi = 0; mi <= 2*M; mi++) {
            int dm;
            double omega;
            mm = mi - M;
            dm = n - mm;                                   /* harmonic index, -H..H */
            omega = 2.0 * M_PI * (f_in + (double)(smallsig ? n : mm) * f0);
            for (e = 0; e < nnz; e++) {
                double gr, gi, cr, ci;
                size_t hi = (size_t)e * (size_t)(H + 1) + (size_t)abs(dm);
                gr = hd->Gmr[hi]; gi = hd->Gmi[hi];
                cr = hd->Cmr[hi]; ci = hd->Cmi[hi];
                if (dm < 0) { gi = -gi; ci = -ci; }        /* conjugate for -h */
                {
                    double er = gr - omega * ci;           /* (g) + j*omega*(c) */
                    double ei = gi + omega * cr;
                    size_t row = (size_t)ni * (size_t)N + (size_t)(hd->rr[e] - 1);
                    size_t col = (size_t)mi * (size_t)N + (size_t)(hd->cc[e] - 1);
                    Ar[row * (size_t)Ntot + col] += er;
                    Ai[row * (size_t)Ntot + col] += ei;
                }
            }
        }
    }
}

/* Solve the conversion matrix at input frequency f_in for a stimulus injected in the
 * 0-th sideband. When `use_src` is set and the netlist supplied an `AC` source, the
 * captured source RHS B_0 is the stimulus; otherwise a unit current is injected at
 * node `inode`. The solution X (all sidebands, length Ntot) is written to Xr/Xi,
 * which the caller allocates. Returns 0 on success, 1 if the matrix is singular. */
static int
pac_solve_at(struct pac_harm *hd, double f0, double f_in, int inode, int use_src,
             double *Xr, double *Xi)
{
    int    N = hd->N, M = hd->M, Ntot = hd->Ntot, rc, j;
    double *Ar, *Ai;

    Ar = TMALLOC(double, (size_t)Ntot * (size_t)Ntot);
    Ai = TMALLOC(double, (size_t)Ntot * (size_t)Ntot);
    pac_build_matrix(hd, f0, f_in, Ar, Ai, 1);

    /* stimulus in the 0-th sideband: netlist AC source RHS, or a unit current */
    memset(Xr, 0, (size_t)Ntot * sizeof(double));
    memset(Xi, 0, (size_t)Ntot * sizeof(double));
    if (use_src && hd->has_src) {
        for (j = 0; j < N; j++) {
            Xr[(size_t)M * (size_t)N + (size_t)j] = hd->B0r[j];
            Xi[(size_t)M * (size_t)N + (size_t)j] = hd->B0i[j];
        }
    } else {
        Xr[(size_t)M * (size_t)N + (size_t)(inode - 1)] = 1.0;
    }
    rc = pss_csolve(Ntot, Ar, Ai, Xr, Xi);
    FREE(Ar); FREE(Ai);
    return rc;
}

#ifdef RFSPICE
/* Enhancement-132: solve the conversion matrix with one RF port DRIVEN (V=1) in the
 * 0-th sideband -- the per-port excitation for periodic S-parameters. The port is a
 * unit voltage source: set its branch-equation RHS to 1 (exactly as .sp's
 * VSRCspupdate does), so V(pos)-V(neg)=1 at that port and 0 at the others (which
 * stay z0-terminated by the g0 shunt already in the Jacobian). Same conversion
 * matrix as pac_solve_at, different RHS. */
static int
psp_solve_port(struct pac_harm *hd, double f0, double f_in, int branch,
               double *Xr, double *Xi)
{
    int    N = hd->N, M = hd->M, Ntot = hd->Ntot, rc;
    double *Ar, *Ai;

    Ar = TMALLOC(double, (size_t)Ntot * (size_t)Ntot);
    Ai = TMALLOC(double, (size_t)Ntot * (size_t)Ntot);
    pac_build_matrix(hd, f0, f_in, Ar, Ai, 1);

    memset(Xr, 0, (size_t)Ntot * sizeof(double));
    memset(Xi, 0, (size_t)Ntot * sizeof(double));
    Xr[(size_t)M * (size_t)N + (size_t)(branch - 1)] = 1.0;
    rc = pss_csolve(Ntot, Ar, Ai, Xr, Xi);
    FREE(Ar); FREE(Ai);
    return rc;
}
#endif

/* pick sideband count M from the requested harmonics; conversion matrix is
 * (2M+1)*N, so cap M so the dense solve stays small. Returns 0 if unusable. */
static int
pac_choose_M(CKTcircuit *ckt, PSSan *job)
{
    int K = ckt->CKTharms, N = job->PSSopMsize, M;
    /* Enhancement-176: honor the requested harmonic count (the old hard cap of
     * 3 sidebands silently truncated the conversion basis -- the +-2 sidebands
     * carried ~10% error with no way to buy accuracy). The dense-solve guard
     * below still bounds the matrix size. */
    M = K - 1;
    if (M < 1)
        return 0;
    while (M > 1 && (2*M + 1) * N > 400)     /* dense-solve guard */
        M--;
    if ((2*M + 1) * N > 400)
        return 0;
    return M;
}

/* Enhancement-121: single-frequency PAC diagnostic, reported for a plain .pss. */
static void
pss_pac_report(CKTcircuit *ckt, PSSan *job)
{
    int    N = job->PSSopMsize, onode = job->PSSoscNode ? job->PSSoscNode->number : 0;
    int    M, e, n;
    double f0 = job->PSSopFreq, f_in;
    struct pac_harm hd;
    double *Xr, *Xi;

    if (onode <= 0 || onode > N || f0 <= 0.0)
        return;
    M = pac_choose_M(ckt, job);
    if (M < 1 || pac_extract_harmonics(ckt, job, M, &hd))
        return;

    f_in = 0.5 * f0;                            /* probe input frequency */
    Xr = TMALLOC(double, hd.Ntot);
    Xi = TMALLOC(double, hd.Ntot);
    if (pac_solve_at(&hd, f0, f_in, onode, 0, Xr, Xi) == 0) {   /* unit-I probe */
        double g0 = 0, c0 = 0, zexp;
        for (e = 0; e < hd.nnz; e++)
            if (hd.rr[e] == onode && hd.cc[e] == onode) {
                g0 = hd.Gmr[(size_t)e * (size_t)(hd.H + 1)];   /* h = 0 */
                c0 = hd.Cmr[(size_t)e * (size_t)(hd.H + 1)];
            }
        zexp = 1.0 / hypot(g0, 2.0 * M_PI * f_in * c0);
        fprintf(stderr, "PAC conversion matrix: f_in = %.6g Hz, %d sidebands, "
                        "unit I at osc node\n", f_in, 2*M + 1);
        for (n = -1; n <= 1; n++) {
            size_t idx = (size_t)(n + M) * (size_t)N + (size_t)(onode - 1);
            fprintf(stderr, "  sideband %+d (%.6g Hz): |V| = %.6g\n",
                    n, f_in + (double)n * f0, hypot(Xr[idx], Xi[idx]));
        }
        fprintf(stderr, "  expected sideband-0 driving-point |Z| = %.6g Ohm "
                        "(linear, from G0/C0)\n", zexp);
    }
    FREE(Xr); FREE(Xi);
    pac_free_harmonics(&hd);
}

/* Enhancement-122/123: PAC frequency sweep (.pac). Extract the Jacobian harmonics
 * once, then sweep the input frequency and, at each point, solve the conversion
 * matrix and emit the response at each requested sideband f_in + k*f0 as a complex
 * plot vs frequency. The stimulus is a netlist-referenced small-signal `AC` source
 * when present (the periodic-AC transfer / conversion gain), else a unit current at
 * the osc node (a driving-point PAC). By default only sideband 0 (the base node
 * names) is emitted; give the `.pac` card a trailing integer `maxsideband` Ksb
 * (after Fstop) -- equivalently the `pac_maxsb` analysis parameter -- to also emit
 * `<node>_usb<k>` / `<node>_lsb<k>` for the upper/lower conversion sidebands
 * k = 1..Ksb. (Note: `pac_maxsb` is an analysis parameter set on the card, not a
 * nutmeg `set` variable.) */
static void
pac_sweep(CKTcircuit *ckt, PSSan *job)
{
    int    N = job->PSSopMsize, onode = job->PSSoscNode ? job->PSSoscNode->number : 0;
    int    M, Ksb, nsb, j, s, k, numNames, nout, error;
    int    stepType = job->PACstepType, np = job->PACpoints;
    double f0 = job->PSSopFreq, fstart = job->PACfStart, fstop = job->PACfStop;
    double freq, mult, linstep;
    struct pac_harm hd;
    double *Xr, *Xi;
    IFuid  freqUid, *nameList = NULL, *outNames = NULL;
    runDesc *pacPlot = NULL;

    if (onode <= 0 || onode > N || f0 <= 0.0 || fstart <= 0.0 ||
        fstop < fstart || np < 1)
        return;
    M = pac_choose_M(ckt, job);
    if (M < 1) {
        fprintf(stderr, "PAC: conversion matrix too large -- sweep skipped\n");
        return;
    }
    if (pac_extract_harmonics(ckt, job, M, &hd))
        return;

    /* number of output sidebands each side (clamped to what the matrix carries) */
    Ksb = job->PACmaxSideband;
    if (Ksb < 0)  Ksb = 0;
    if (Ksb > M)  Ksb = M;
    nsb = 2 * Ksb + 1;

    /* build the output name list: base node names for sideband 0, plus
     * <node>_usb<k> / <node>_lsb<k> for the upper/lower conversion sidebands. */
    error = CKTnames(ckt, &numNames, &nameList);
    if (error || numNames != N) { pac_free_harmonics(&hd); FREE(nameList); return; }
    nout = numNames * nsb;
    outNames = TMALLOC(IFuid, nout);
    for (s = 0; s < nsb; s++) {
        k = s - Ksb;
        for (j = 0; j < numNames; j++) {
            if (k == 0) {
                outNames[s * numNames + j] = nameList[j];   /* reuse base UID */
            } else {
                char nm[256];
                IFuid uid;
                (void) snprintf(nm, sizeof(nm), "%s_%csb%d",
                                (char *) nameList[j], (k > 0) ? 'u' : 'l', abs(k));
                if (SPfrontEnd->IFnewUid(ckt, &uid, NULL, nm, UID_OTHER, NULL))
                    uid = nameList[j];                      /* fallback on clash */
                outNames[s * numNames + j] = uid;
            }
        }
    }

    SPfrontEnd->IFnewUid(ckt, &freqUid, NULL, "frequency", UID_OTHER, NULL);
    error = SPfrontEnd->OUTpBeginPlot(ckt, ckt->CKTcurJob, "PAC Analysis",
                                      freqUid, IF_REAL, nout, outNames,
                                      IF_COMPLEX, &pacPlot);
    tfree(nameList);
    tfree(outNames);
    if (error) { pac_free_harmonics(&hd); return; }
    if (stepType != 0)      /* dec / oct -> log frequency axis */
        SPfrontEnd->OUTattributes(pacPlot, NULL, OUT_SCALE_LOG, NULL);

    Xr = TMALLOC(double, hd.Ntot);
    Xi = TMALLOC(double, hd.Ntot);
    mult    = (stepType == 1) ? pow(10.0, 1.0 / np) :
              (stepType == 2) ? pow(2.0,  1.0 / np) : 0.0;
    linstep = (np > 1) ? (fstop - fstart) / (np - 1) : 0.0;

    fprintf(stderr, "PAC sweep: %s from %.6g to %.6g Hz (%d pts/%s) around "
                    "f0 = %.6g Hz; stimulus: %s; %d sideband%s\n",
            (stepType == 1) ? "dec" : (stepType == 2) ? "oct" : "lin",
            fstart, fstop, np,
            (stepType == 0) ? "total" : (stepType == 1) ? "decade" : "octave", f0,
            hd.has_src ? "netlist AC source" : "unit I at osc node",
            nsb, (nsb == 1) ? "" : "s");

    for (freq = fstart; freq <= fstop * (1.0 + 1e-9); ) {
        if (pac_solve_at(&hd, f0, freq, onode, 1, Xr, Xi) == 0) {
            IFvalue freqData, valueData;
            IFcomplex *data = TMALLOC(IFcomplex, nout);
            freqData.rValue = freq;
            valueData.v.numValue = nout;
            valueData.v.vec.cVec = data;
            for (s = 0; s < nsb; s++) {
                size_t blk = (size_t)(s - Ksb + M) * (size_t)N;   /* sideband block */
                for (j = 0; j < numNames; j++) {
                    data[s * numNames + j].real = Xr[blk + (size_t)j];
                    data[s * numNames + j].imag = Xi[blk + (size_t)j];
                }
            }
            SPfrontEnd->OUTpData(pacPlot, &freqData, &valueData);
            FREE(data);
        }
        if (stepType == 0) { if (np <= 1) break; freq += linstep; }
        else               { freq *= mult; }
    }

    SPfrontEnd->OUTendPlot(pacPlot);
    FREE(Xr); FREE(Xi);
    pac_free_harmonics(&hd);
}


/* Enhancement-124: solve the ADJOINT conversion system Hᵀ Psi = e_{out,0}. Psi
 * then holds, for every (node j, sideband k), the transfer from a unit injection
 * at (j,k) to the output at sideband 0 -- the conversion transimpedance the noise
 * folding needs. Assembles Hᵀ (the transpose of the pac_solve_at matrix) and puts a
 * unit at the output node in the 0-th sideband. Returns 0 on success, 1 if singular. */
static int
pac_solve_adjoint(struct pac_harm *hd, double f0, double f_in, int outNode,
                  double *Psr, double *Psi)
{
    int    N = hd->N, M = hd->M, H = hd->H, nnz = hd->nnz, Ntot = hd->Ntot;
    int    ni, mi, n, mm, e, rc;
    double *Ar, *Ai;

    Ar = TMALLOC(double, (size_t)Ntot * (size_t)Ntot);
    Ai = TMALLOC(double, (size_t)Ntot * (size_t)Ntot);
    memset(Ar, 0, (size_t)Ntot * (size_t)Ntot * sizeof(double));
    memset(Ai, 0, (size_t)Ntot * (size_t)Ntot * sizeof(double));
    for (ni = 0; ni <= 2*M; ni++) {
        n = ni - M;
        for (mi = 0; mi <= 2*M; mi++) {
            int dm;
            double omega;
            mm = mi - M;
            dm = n - mm;
            /* small-signal reactive multiplier: ROW (output) sideband frequency
             * (see pac_build_matrix) -- the transpose below swaps storage, not
             * the roles of n and m in the operator being transposed. */
            omega = 2.0 * M_PI * (f_in + (double)n * f0);
            for (e = 0; e < nnz; e++) {
                double gr, gi, cr, ci;
                size_t hi = (size_t)e * (size_t)(H + 1) + (size_t)abs(dm);
                gr = hd->Gmr[hi]; gi = hd->Gmi[hi];
                cr = hd->Cmr[hi]; ci = hd->Cmi[hi];
                if (dm < 0) { gi = -gi; ci = -ci; }
                {
                    double er = gr - omega * ci;
                    double ei = gi + omega * cr;
                    size_t row = (size_t)ni * (size_t)N + (size_t)(hd->rr[e] - 1);
                    size_t col = (size_t)mi * (size_t)N + (size_t)(hd->cc[e] - 1);
                    Ar[col * (size_t)Ntot + row] += er;   /* transpose: [col][row] */
                    Ai[col * (size_t)Ntot + row] += ei;
                }
            }
        }
    }

    memset(Psr, 0, (size_t)Ntot * sizeof(double));
    memset(Psi, 0, (size_t)Ntot * sizeof(double));
    Psr[(size_t)M * (size_t)N + (size_t)(outNode - 1)] = 1.0;
    rc = pss_csolve(Ntot, Ar, Ai, Psr, Psi);
    FREE(Ar); FREE(Ai);
    return rc;
}


/* Enhancement-124: periodic noise (.pnoise). Runs off the retained operating point:
 * folds every device's noise through the conversion-matrix adjoint over all
 * sidebands to get the output noise spectrum. The device noise routines
 * (DEVnoise/NevalSrc, OSDI load_noise) compute S*|dTransimp|^2 reading the
 * transimpedance from CKTrhs/CKTirhs -- so loading the sideband-k adjoint into
 * CKTrhs and summing over k = -M..M folds the noise exactly. A local NOISEAN job
 * gives those routines their expected context. For a linear (block-diagonal)
 * circuit only sideband 0 contributes, so the result reduces to ordinary .noise. */
static void
pnoise_sweep(CKTcircuit *ckt, PSSan *job)
{
    int    N = job->PSSopMsize, ns = job->PSSopNumStates;
    int    outNode = job->PnOutNode ? job->PnOutNode->number : 0;
    int    M, i, j, k, np = job->PACpoints, stepType = job->PACstepType, error;
    double f0 = job->PSSopFreq, fstart = job->PACfStart, fstop = job->PACfStop;
    double freq, mult, linstep;
    struct pac_harm hd;
    double *Psr, *Psi, *Xr, *Xi;
    NOISEAN nj;
    Ndata   data;
    JOB    *oldJob;
    IFuid   freqUid, nlist[2];
    runDesc *plot = NULL;
    /* Enhancement-193: honor the `sqrnoise` control variable, exactly like
     * .noise (noisean.c): default (unset) reports the noise spectral density in
     * V/sqrt(Hz) / A/sqrt(Hz); `set sqrnoise` reports the squared V^2/Hz form.
     * Before E-193 pnoise always emitted the squared density and ignored the
     * variable, contradicting the manual and diverging from .noise. */
    int     pn_sqr = cp_getvar("sqrnoise", CP_BOOL, NULL, 0);

    if (outNode <= 0 || outNode > N || f0 <= 0.0 || fstart <= 0.0 ||
        fstop < fstart || np < 1)
        return;
    M = pac_choose_M(ckt, job);
    if (M < 1) {
        fprintf(stderr, "PNOISE: conversion matrix too large -- sweep skipped\n");
        return;
    }
    if (pac_extract_harmonics(ckt, job, M, &hd))
        return;

    /* set the device bias to the (sample-0) operating point so each noise PSD
     * (conductances, dc currents) is evaluated at the periodic operating point. */
    for (i = 1; i <= N; i++)
        ckt->CKTrhsOld[i] = job->PSSopVoltages[(i - 1)];
    ckt->CKTrhsOld[0] = 0.0;
    if (ns > 0)
        memcpy(ckt->CKTstate0, job->PSSopStates, (size_t)ns * sizeof(double));
    ckt->CKTmode = (ckt->CKTmode & MODEUIC) | MODEDCOP | MODEINITSMSIG;
    CKTload(ckt);

    /* output plot (onoise/inoise spectrum vs frequency), opened while CKTcurJob is
     * still the persistent PSS job. */
    SPfrontEnd->IFnewUid(ckt, &freqUid, NULL, "frequency", UID_OTHER, NULL);
    SPfrontEnd->IFnewUid(ckt, &nlist[0], NULL, "onoise_spectrum", UID_OTHER, NULL);
    SPfrontEnd->IFnewUid(ckt, &nlist[1], NULL, "inoise_spectrum", UID_OTHER, NULL);
    error = SPfrontEnd->OUTpBeginPlot(ckt, ckt->CKTcurJob, "PNoise Analysis",
                                      freqUid, IF_REAL, 2, nlist, IF_REAL, &plot);
    if (error) { pac_free_harmonics(&hd); return; }
    if (stepType != 0)
        SPfrontEnd->OUTattributes(plot, NULL, OUT_SCALE_LOG, NULL);

    /* a minimal NOISEAN context for the device noise routines (they cast
     * CKTcurJob to NOISEAN* and read NStpsSm / NstartFreq). */
    memset(&nj, 0, sizeof(nj));
    nj.output = job->PnOutNode;
    nj.outputRef = job->PnOutNode;
    nj.input = job->PnInSrc;
    nj.NstartFreq = fstart;
    nj.NstopFreq = fstop;
    nj.NnumSteps = np;
    nj.NstpType = stepType;
    nj.NStpsSm = 0;                 /* no per-device summary vectors */
    nj.JOBname = "pnoise";
    memset(&data, 0, sizeof(data));
    data.prtSummary = FALSE;        /* keep the routines from writing outpVector */

    oldJob = ckt->CKTcurJob;
    ckt->CKTcurJob = (JOB *) &nj;

    /* let each device set up its noise state (a no-op naming pass with NStpsSm=0) */
    for (i = 0; i < DEVmaxnum; i++)
        if (DEVices[i] && DEVices[i]->DEVnoise && ckt->CKThead[i]) {
            double dummy = 0.0;
            DEVices[i]->DEVnoise(N_DENS, N_OPEN, ckt->CKThead[i], ckt, &data, &dummy);
        }

    Psr = TMALLOC(double, hd.Ntot);  Psi = TMALLOC(double, hd.Ntot);
    Xr  = TMALLOC(double, hd.Ntot);  Xi  = TMALLOC(double, hd.Ntot);
    mult    = (stepType == 1) ? pow(10.0, 1.0 / np) :
              (stepType == 2) ? pow(2.0,  1.0 / np) : 0.0;
    linstep = (np > 1) ? (fstop - fstart) / (np - 1) : 0.0;

    fprintf(stderr, "PNOISE sweep: %s from %.6g to %.6g Hz around f0 = %.6g Hz; "
                    "output node %d; folding %d sidebands%s\n",
            (stepType == 1) ? "dec" : (stepType == 2) ? "oct" : "lin",
            fstart, fstop, f0, outNode, 2*M + 1,
            job->PSSpnCyclo ? "; cyclostationary" : "");

    if (job->PSSpnCyclo) {
        /* Enhancement-178: EXACT separable cyclostationary folding.
         *
         * E-126's identity onoise = (1/P) Sum_s S(t_s)|A_s|^2 is exact only for
         * frequency-FLAT (modulated-white) sources: it cannot see that noise
         * folded from sideband q ORIGINATES at |f + q*f0| where a colored
         * source (flicker 1/f, noise_table) has a different density (E-177).
         * For the physical separable model S(t, f) = m(t)^2 * g(f) (a colored
         * stationary process amplitude-modulated along the orbit) the exact
         * output noise is
         *
         *   onoise(f) = Sum_g Sum_q | B_q(g) |^2 * g_g(|f + q*f0|),
         *   B_q(g)    = (1/P) Sum_s m_g(t_s) * dA_s(g) * e^{+j q th_s},
         *   dA_s(g)   = the generator's transimpedance difference of
         *               A-_s(j) = Sum_k Psi_k(j) e^{-j k th_s}.
         *
         * (Stationary limit m = const: B_q = m*dPsi_q, recovering the E-177
         * stationary sum exactly; flat limit g = const: Parseval collapses the
         * q-sum back to E-126's time-domain identity. This path therefore
         * strictly generalizes both.)
         *
         * The per-generator amplitudes are recovered WITHOUT any device-API
         * change through the noise-summary machinery (data->prtSummary fills
         * data->outpVector with one density per generator) plus load
         * POLARIZATION against a fixed reference load R = Psi_0:
         * four sweeps with loads A+-R, A+-jR give the complex
         * S_g(t_s)*dA_s(g)*conj(dR(g)) per generator, a fifth (load R) gives
         * S_g(t_s)*|dR(g)|^2, and c_g,s = y/sqrt(z) equals
         * sqrt(S_g(t_s)) * dA_s(g) * (unit phasor that cancels in |B_q|^2).
         * The spectral shape g_g is MEASURED pointwise at the needed
         * frequencies |f + q*f0| (no 1/f^EF assumption -- noise_table works),
         * at several probe biases so a generator quiet at one phase still gets
         * its shape from an active phase.
         *
         * Generators whose measured shape is flat are routed through the
         * original time-domain identity (exact for ANY quadratic form,
         * including NevalSrc2-correlated pairs); colored generators through
         * the B_q machinery (exact for the rank-1 node-pair sources that all
         * colored generators are in practice). A colored generator invisible
         * to the reference load falls back to the flat identity (the old
         * behavior) rather than being dropped. */
        long   P = job->PSSopPoints, s;
        int    Nf = 0, fi, c, G, g, qi;
        int    Qmax = 2 * M, nq = 2 * Qmax + 1;
        int    NB = 8;                      /* shape-probe biases */
        double *freqs, *onz, *Pr_all, *Pi_all;
        Ndata   d2;
        NOISEAN nj2;
        int    *is_total;
        double *swv[5];
        double *Bqr, *Bqi, *Dflat, *zsum, *probe;
        double *Lr, *Li;
        double  dummy_dens;

        for (freq = fstart; freq <= fstop * (1.0 + 1e-9); ) {   /* count points */
            Nf++;
            if (stepType == 0) { if (np <= 1) break; freq += linstep; }
            else               { freq *= mult; }
        }
        freqs  = TMALLOC(double, Nf);
        onz    = TMALLOC(double, Nf);
        Pr_all = TMALLOC(double, (size_t)Nf * (size_t)hd.Ntot);
        Pi_all = TMALLOC(double, (size_t)Nf * (size_t)hd.Ntot);
        c = 0;
        for (freq = fstart; freq <= fstop * (1.0 + 1e-9); ) {   /* fill + adjoints */
            freqs[c] = freq;  onz[c] = 0.0;
            if (pac_solve_adjoint(&hd, f0, freq, outNode, Psr, Psi) != 0) {
                memset(Psr, 0, (size_t)hd.Ntot * sizeof(double));
                memset(Psi, 0, (size_t)hd.Ntot * sizeof(double));
            }
            memcpy(Pr_all + (size_t)c * (size_t)hd.Ntot, Psr, (size_t)hd.Ntot * sizeof(double));
            memcpy(Pi_all + (size_t)c * (size_t)hd.Ntot, Psi, (size_t)hd.Ntot * sizeof(double));
            c++;
            if (stepType == 0) { if (np <= 1) break; freq += linstep; }
            else               { freq *= mult; }
        }

        /* --- generator naming pass: one summary slot per noise generator --- */
        nj2 = nj;
        nj2.NStpsSm = 1;
        memset(&d2, 0, sizeof d2);
        ckt->CKTcurJob = (JOB *) &nj2;
        for (i = 0; i < DEVmaxnum; i++)
            if (DEVices[i] && DEVices[i]->DEVnoise && ckt->CKThead[i]) {
                dummy_dens = 0.0;
                DEVices[i]->DEVnoise(N_DENS, N_OPEN, ckt->CKThead[i], ckt, &d2, &dummy_dens);
            }
        G = d2.numPlots;
        is_total = TMALLOC(int, (G > 0) ? G : 1);
        for (g = 0; g < G; g++) {
            /* a slot whose name is a proper prefix of the preceding slot's name
             * is the family TOTAL ("onoise_R1" after "onoise_R1_1overf"):
             * totals are redundant sums and must not be folded again */
            is_total[g] = 0;
            if (g > 0) {
                const char *nm = (const char *) d2.namelist[g];
                const char *pm = (const char *) d2.namelist[g - 1];
                size_t ln = strlen(nm);
                if (strncmp(nm, pm, ln) == 0 && strlen(pm) > ln)
                    is_total[g] = 1;
            }
        }
        d2.outpVector = TMALLOC(double, (G > 0) ? G : 1);
        d2.prtSummary = TRUE;
        d2.delFreq = 0.0;
        for (i = 0; i < 5; i++)
            swv[i] = TMALLOC(double, (G > 0) ? G : 1);
        Bqr   = TMALLOC(double, (size_t)Nf * (size_t)G * (size_t)nq);
        Bqi   = TMALLOC(double, (size_t)Nf * (size_t)G * (size_t)nq);
        Dflat = TMALLOC(double, (size_t)Nf * (size_t)G);
        zsum  = TMALLOC(double, (size_t)Nf * (size_t)G);
        probe = TMALLOC(double, (size_t)NB * (size_t)Nf * (size_t)G * (size_t)nq);
        memset(Bqr, 0, (size_t)Nf * (size_t)G * (size_t)nq * sizeof(double));
        memset(Bqi, 0, (size_t)Nf * (size_t)G * (size_t)nq * sizeof(double));
        memset(Dflat, 0, (size_t)Nf * (size_t)G * sizeof(double));
        memset(zsum, 0, (size_t)Nf * (size_t)G * sizeof(double));
        memset(probe, 0, (size_t)NB * (size_t)Nf * (size_t)G * (size_t)nq * sizeof(double));
        Lr = TMALLOC(double, N + 1);
        Li = TMALLOC(double, N + 1);

        /* --- main collection loop: per sample, per frequency, five sweeps --- */
        for (s = 0; s < P; s++) {
            double ang0 = 2.0 * M_PI * (double)s / (double)P;
            for (i = 1; i <= N; i++)
                ckt->CKTrhsOld[i] = job->PSSopVoltages[(i - 1) + s * N];
            ckt->CKTrhsOld[0] = 0.0;
            if (ns > 0)
                memcpy(ckt->CKTstate0, job->PSSopStates + (size_t)s * (size_t)ns,
                       (size_t)ns * sizeof(double));
            ckt->CKTmode = (ckt->CKTmode & MODEUIC) | MODEDCOP | MODEINITSMSIG;
            CKTload(ckt);
            for (fi = 0; fi < Nf; fi++) {
                double *pr = Pr_all + (size_t)fi * (size_t)hd.Ntot;
                double *pi = Pi_all + (size_t)fi * (size_t)hd.Ntot;
                int sweep;
                for (j = 1; j <= N; j++) {   /* A-_s(j) = Sum_k Psi_k(j) e^{-jk th} */
                    double ar = 0.0, ai = 0.0;
                    for (k = -M; k <= M; k++) {
                        size_t idx = (size_t)(k + M) * (size_t)N + (size_t)(j - 1);
                        double cs = cos((double)k * ang0), sn = sin((double)k * ang0);
                        ar += pr[idx] * cs + pi[idx] * sn;    /* e^{-jk*th} */
                        ai += pi[idx] * cs - pr[idx] * sn;
                    }
                    Lr[j] = ar;  Li[j] = ai;
                }
                for (sweep = 0; sweep < 5; sweep++) {
                    size_t r0 = (size_t)M * (size_t)N;        /* Psi_0 block = R */
                    for (j = 1; j <= N; j++) {
                        double rr = pr[r0 + (size_t)(j - 1)];
                        double ri = pi[r0 + (size_t)(j - 1)];
                        switch (sweep) {
                        case 0: ckt->CKTrhs[j] = Lr[j] + rr;  ckt->CKTirhs[j] = Li[j] + ri;  break;
                        case 1: ckt->CKTrhs[j] = Lr[j] - rr;  ckt->CKTirhs[j] = Li[j] - ri;  break;
                        case 2: ckt->CKTrhs[j] = Lr[j] - ri;  ckt->CKTirhs[j] = Li[j] + rr;  break; /* A + jR */
                        case 3: ckt->CKTrhs[j] = Lr[j] + ri;  ckt->CKTirhs[j] = Li[j] - rr;  break; /* A - jR */
                        default: ckt->CKTrhs[j] = rr;         ckt->CKTirhs[j] = ri;          break; /* R */
                        }
                    }
                    ckt->CKTrhs[0] = 0.0;  ckt->CKTirhs[0] = 0.0;
                    d2.freq = freqs[fi];  d2.outNumber = 0;
                    dummy_dens = 0.0;
                    for (i = 0; i < DEVmaxnum; i++)
                        if (DEVices[i] && DEVices[i]->DEVnoise && ckt->CKThead[i])
                            DEVices[i]->DEVnoise(N_DENS, N_CALC, ckt->CKThead[i],
                                                 ckt, &d2, &dummy_dens);
                    memcpy(swv[sweep], d2.outpVector, (size_t)G * sizeof(double));
                }
                for (g = 0; g < G; g++) {
                    double dp, dm, djp, djm, z, qa;
                    size_t bfg = ((size_t)fi * (size_t)G + (size_t)g) * (size_t)nq;
                    if (is_total[g])
                        continue;
                    dp = swv[0][g];  dm = swv[1][g];
                    djp = swv[2][g]; djm = swv[3][g];
                    z = swv[4][g];
                    qa = 0.5 * (dp + dm) - z;                 /* S_g * Q_g(A-_s) */
                    Dflat[(size_t)fi * (size_t)G + (size_t)g] += qa;
                    if (z > 1e-280) {
                        double yr = 0.25 * (dp - dm);          /* S*Re(dA dR*) */
                        double yi = 0.25 * (djp - djm);        /* S*Im(dA dR*) */
                        double sz = sqrt(z);
                        double cr = yr / sz, ci = yi / sz;     /* c_g,s */
                        zsum[(size_t)fi * (size_t)G + (size_t)g] += z;
                        for (qi = 0; qi < nq; qi++) {
                            double aq = (double)(qi - Qmax) * ang0;
                            double cq = cos(aq), sq = sin(aq);
                            Bqr[bfg + (size_t)qi] += cr * cq - ci * sq;   /* c * e^{+jq th} */
                            Bqi[bfg + (size_t)qi] += cr * sq + ci * cq;
                        }
                    }
                }
            }
        }

        /* --- spectral-shape probes: S_g(t_b, |f + q f0|) with load R --- */
        {
            int b;
            for (b = 0; b < NB; b++) {
                long sb = (long)b * P / NB;
                for (i = 1; i <= N; i++)
                    ckt->CKTrhsOld[i] = job->PSSopVoltages[(i - 1) + sb * N];
                ckt->CKTrhsOld[0] = 0.0;
                if (ns > 0)
                    memcpy(ckt->CKTstate0, job->PSSopStates + (size_t)sb * (size_t)ns,
                           (size_t)ns * sizeof(double));
                ckt->CKTmode = (ckt->CKTmode & MODEUIC) | MODEDCOP | MODEINITSMSIG;
                CKTload(ckt);
                for (fi = 0; fi < Nf; fi++) {
                    double *pr = Pr_all + (size_t)fi * (size_t)hd.Ntot;
                    double *pi = Pi_all + (size_t)fi * (size_t)hd.Ntot;
                    size_t r0 = (size_t)M * (size_t)N;
                    for (j = 1; j <= N; j++) {
                        ckt->CKTrhs[j]  = pr[r0 + (size_t)(j - 1)];
                        ckt->CKTirhs[j] = pi[r0 + (size_t)(j - 1)];
                    }
                    ckt->CKTrhs[0] = 0.0;  ckt->CKTirhs[0] = 0.0;
                    for (qi = 0; qi < nq; qi++) {
                        double fq = fabs(freqs[fi] + (double)(qi - Qmax) * f0);
                        if (fq == 0.0)
                            fq = freqs[fi];
                        d2.freq = fq;  d2.outNumber = 0;
                        dummy_dens = 0.0;
                        for (i = 0; i < DEVmaxnum; i++)
                            if (DEVices[i] && DEVices[i]->DEVnoise && ckt->CKThead[i])
                                DEVices[i]->DEVnoise(N_DENS, N_CALC, ckt->CKThead[i],
                                                     ckt, &d2, &dummy_dens);
                        for (g = 0; g < G; g++)
                            probe[(((size_t)b * (size_t)Nf + (size_t)fi) * (size_t)G
                                   + (size_t)g) * (size_t)nq + (size_t)qi] = d2.outpVector[g];
                    }
                }
            }
        }

        /* --- combine: flat slots via the time-domain identity, colored slots
         *     via Sum_q |B_q|^2 * r(q) with the measured shape ratio --- */
        for (fi = 0; fi < Nf; fi++) {
            for (g = 0; g < G; g++) {
                double zref = 0.0, on_g;
                int    b, bbest = 0, colored = 0;
                size_t bfg = ((size_t)fi * (size_t)G + (size_t)g) * (size_t)nq;
                if (is_total[g])
                    continue;
                for (b = 0; b < NB; b++) {                    /* most active probe bias */
                    double zr = probe[(((size_t)b * (size_t)Nf + (size_t)fi) * (size_t)G
                                       + (size_t)g) * (size_t)nq + (size_t)Qmax];
                    if (zr > zref) { zref = zr; bbest = b; }
                }
                if (zref > 0.0) {
                    for (qi = 0; qi < nq; qi++) {
                        double rq = probe[(((size_t)bbest * (size_t)Nf + (size_t)fi) * (size_t)G
                                           + (size_t)g) * (size_t)nq + (size_t)qi] / zref;
                        if (fabs(rq - 1.0) > 1e-6) { colored = 1; break; }
                    }
                }
                if (colored && zsum[(size_t)fi * (size_t)G + (size_t)g] > 0.0) {
                    on_g = 0.0;
                    for (qi = 0; qi < nq; qi++) {
                        double rq = probe[(((size_t)bbest * (size_t)Nf + (size_t)fi) * (size_t)G
                                           + (size_t)g) * (size_t)nq + (size_t)qi] / zref;
                        double br = Bqr[bfg + (size_t)qi] / (double)P;
                        double bi = Bqi[bfg + (size_t)qi] / (double)P;
                        on_g += (br * br + bi * bi) * rq;
                    }
                } else {
                    /* flat generator, or colored one invisible to the reference
                     * load: the (any-rank exact / legacy) time-domain identity */
                    on_g = Dflat[(size_t)fi * (size_t)G + (size_t)g] / (double)P;
                }
                onz[fi] += on_g;
            }
        }

        for (fi = 0; fi < Nf; fi++) {   /* gain, output */
            double onoise = onz[fi], gain2 = 1.0, gsi;
            IFvalue refVal, valData;
            double out[2];
            if (hd.has_src && pac_solve_at(&hd, f0, freqs[fi], outNode, 1, Xr, Xi) == 0) {
                size_t oidx = (size_t)M * (size_t)N + (size_t)(outNode - 1);
                gain2 = Xr[oidx] * Xr[oidx] + Xi[oidx] * Xi[oidx];
            }
            gsi = 1.0 / MAX(gain2, N_MINGAIN);
            out[0] = onoise;  out[1] = onoise * gsi;
            if (!pn_sqr) {                          /* E-193: V/sqrt(Hz) by default */
                out[0] = sqrt(out[0]);  out[1] = sqrt(out[1]);
            }
            refVal.rValue = freqs[fi];
            valData.v.numValue = 2;  valData.v.vec.rVec = out;
            SPfrontEnd->OUTpData(plot, &refVal, &valData);
        }
        ckt->CKTcurJob = (JOB *) &nj;
        for (i = 0; i < 5; i++) FREE(swv[i]);
        FREE(is_total); FREE(d2.outpVector); FREE(d2.namelist);
        FREE(Bqr); FREE(Bqi); FREE(Dflat); FREE(zsum); FREE(probe);
        FREE(Lr); FREE(Li);
        FREE(freqs); FREE(onz); FREE(Pr_all); FREE(Pi_all);
    } else
    for (freq = fstart; freq <= fstop * (1.0 + 1e-9); ) {
        double onoise = 0.0, gain2 = 1.0, gsi;

        data.delFreq = 0.0;         /* density only -- we do not integrate here */
        data.prtSummary = FALSE;

        /* transfer from every (node, sideband) to the output at sideband 0 */
        if (pac_solve_adjoint(&hd, f0, freq, outNode, Psr, Psi) == 0) {
            for (k = -M; k <= M; k++) {
                double dens = 0.0;
                size_t blk = (size_t)(k + M) * (size_t)N;
                /* Enhancement-177: the sideband-k adjoint carries noise that
                 * ORIGINATES at the physical frequency |freq + k*f0| -- a
                 * frequency-dependent source PSD (flicker 1/f, noise_table)
                 * must be evaluated THERE, not at the output frequency.
                 * (White sources are flat, and LTI circuits have no k != 0
                 * transfer, which is why every earlier check passed.) */
                data.freq = fabs(freq + (double)k * f0);
                if (data.freq == 0.0)
                    data.freq = freq;   /* exact-hit guard (1/f at dc) */
                for (j = 1; j <= N; j++) {
                    ckt->CKTrhs[j]  = Psr[blk + (size_t)(j - 1)];
                    ckt->CKTirhs[j] = Psi[blk + (size_t)(j - 1)];
                }
                ckt->CKTrhs[0] = 0.0;  ckt->CKTirhs[0] = 0.0;
                for (i = 0; i < DEVmaxnum; i++)
                    if (DEVices[i] && DEVices[i]->DEVnoise && ckt->CKThead[i])
                        DEVices[i]->DEVnoise(N_DENS, N_CALC, ckt->CKThead[i],
                                             ckt, &data, &dens);
                onoise += dens;                 /* sum device noise over sidebands */
            }
        }

        /* input-referred: divide by the source->output conversion gain (sideband 0) */
        if (hd.has_src && pac_solve_at(&hd, f0, freq, outNode, 1, Xr, Xi) == 0) {
            size_t oidx = (size_t)M * (size_t)N + (size_t)(outNode - 1);
            gain2 = Xr[oidx] * Xr[oidx] + Xi[oidx] * Xi[oidx];
        }
        gsi = 1.0 / MAX(gain2, N_MINGAIN);

        {
            IFvalue refVal, valData;
            double out[2];
            out[0] = onoise;                    /* output noise density (V^2/Hz) */
            out[1] = onoise * gsi;              /* input-referred density */
            if (!pn_sqr) {                      /* E-193: V/sqrt(Hz) by default */
                out[0] = sqrt(out[0]);  out[1] = sqrt(out[1]);
            }
            refVal.rValue = freq;
            valData.v.numValue = 2;
            valData.v.vec.rVec = out;
            SPfrontEnd->OUTpData(plot, &refVal, &valData);
        }

        if (stepType == 0) { if (np <= 1) break; freq += linstep; }
        else               { freq *= mult; }
    }

    SPfrontEnd->OUTendPlot(plot);
    ckt->CKTcurJob = oldJob;
    FREE(Psr); FREE(Psi); FREE(Xr); FREE(Xi);
    pac_free_harmonics(&hd);
}


/* Enhancement-125: periodic transfer function (.pxf). The adjoint counterpart of
 * .pac: solve Hᵀ Ψ = e_{out,0} once per frequency and dot each sideband block of Ψ
 * with the netlist AC-source pattern B_0 to get the transfer from the input to the
 * fixed output at each sideband, xf_k = Σ_j Ψ_k(j)·B0(j). By the identity
 * (H⁻¹B)_out = (H⁻ᵀe_out)ᵀB, the sideband-0 transfer equals the PAC response at the
 * output exactly -- the reciprocity cross-check. Emits xf (sideband 0) plus
 * xf_usb<k>/xf_lsb<k> conversion transfers as a complex plot vs frequency. */
static void
pxf_sweep(CKTcircuit *ckt, PSSan *job)
{
    int    N = job->PSSopMsize, outNode = job->PxOutNode ? job->PxOutNode->number : 0;
    int    M, Ksb, nsb, s, k, j, error, stepType = job->PACstepType, np = job->PACpoints;
    double f0 = job->PSSopFreq, fstart = job->PACfStart, fstop = job->PACfStop;
    double freq, mult, linstep;
    struct pac_harm hd;
    double *Psr, *Psi;
    IFuid  freqUid, *outNames = NULL;
    runDesc *plot = NULL;
    char nm[64];

    if (outNode <= 0 || outNode > N || f0 <= 0.0 || fstart <= 0.0 ||
        fstop < fstart || np < 1)
        return;
    M = pac_choose_M(ckt, job);
    if (M < 1) {
        fprintf(stderr, "PXF: conversion matrix too large -- sweep skipped\n");
        return;
    }
    if (pac_extract_harmonics(ckt, job, M, &hd))
        return;
    if (!hd.has_src) {
        fprintf(stderr, "PXF: no netlist AC source found -- give the input source an "
                        "AC value; sweep skipped\n");
        pac_free_harmonics(&hd);
        return;
    }

    Ksb = job->PACmaxSideband;
    if (Ksb < 0)  Ksb = 0;
    if (Ksb > M)  Ksb = M;
    nsb = 2 * Ksb + 1;

    /* one transfer vector per output sideband: xf (sb0), xf_usb<k>, xf_lsb<k> */
    outNames = TMALLOC(IFuid, nsb);
    for (s = 0; s < nsb; s++) {
        k = s - Ksb;
        if (k == 0)
            (void) snprintf(nm, sizeof(nm), "xf");
        else
            (void) snprintf(nm, sizeof(nm), "xf_%csb%d", (k > 0) ? 'u' : 'l', abs(k));
        SPfrontEnd->IFnewUid(ckt, &outNames[s], NULL, nm, UID_OTHER, NULL);
    }

    SPfrontEnd->IFnewUid(ckt, &freqUid, NULL, "frequency", UID_OTHER, NULL);
    error = SPfrontEnd->OUTpBeginPlot(ckt, ckt->CKTcurJob, "PXF Analysis",
                                      freqUid, IF_REAL, nsb, outNames,
                                      IF_COMPLEX, &plot);
    tfree(outNames);
    if (error) { pac_free_harmonics(&hd); return; }
    if (stepType != 0)
        SPfrontEnd->OUTattributes(plot, NULL, OUT_SCALE_LOG, NULL);

    Psr = TMALLOC(double, hd.Ntot);
    Psi = TMALLOC(double, hd.Ntot);
    mult    = (stepType == 1) ? pow(10.0, 1.0 / np) :
              (stepType == 2) ? pow(2.0,  1.0 / np) : 0.0;
    linstep = (np > 1) ? (fstop - fstart) / (np - 1) : 0.0;

    fprintf(stderr, "PXF sweep: %s from %.6g to %.6g Hz around f0 = %.6g Hz; "
                    "output node %d; %d sideband%s (adjoint)\n",
            (stepType == 1) ? "dec" : (stepType == 2) ? "oct" : "lin",
            fstart, fstop, f0, outNode, nsb, (nsb == 1) ? "" : "s");

    for (freq = fstart; freq <= fstop * (1.0 + 1e-9); ) {
        if (pac_solve_adjoint(&hd, f0, freq, outNode, Psr, Psi) == 0) {
            IFvalue freqData, valData;
            IFcomplex *data = TMALLOC(IFcomplex, nsb);
            freqData.rValue = freq;
            valData.v.numValue = nsb;
            valData.v.vec.cVec = data;
            for (s = 0; s < nsb; s++) {
                size_t blk = (size_t)(s - Ksb + M) * (size_t)N;
                double xr = 0.0, xi = 0.0;   /* xf_k = sum_j Psi_k(j) * B0(j) */
                for (j = 0; j < N; j++) {
                    xr += Psr[blk + (size_t)j] * hd.B0r[j] - Psi[blk + (size_t)j] * hd.B0i[j];
                    xi += Psr[blk + (size_t)j] * hd.B0i[j] + Psi[blk + (size_t)j] * hd.B0r[j];
                }
                data[s].real = xr;
                data[s].imag = xi;
            }
            SPfrontEnd->OUTpData(plot, &freqData, &valData);
            FREE(data);
        }
        if (stepType == 0) { if (np <= 1) break; freq += linstep; }
        else               { freq *= mult; }
    }

    SPfrontEnd->OUTendPlot(plot);
    FREE(Psr); FREE(Psi);
    pac_free_harmonics(&hd);
}


#ifdef RFSPICE
/* Enhancement-132: periodic S-parameters (.psp). After PSS, excite each RF port in
 * turn with a unit current in the 0-th sideband through the conversion matrix, read
 * the per-sideband port waves a/b (Kurosawa power waves, matching .sp's convention),
 * and form the periodic scattering matrix S^(k) = B^(k) * A^-1 at each swept input
 * frequency.  S = B*A^-1 is invariant to the excitation basis, so unit-current
 * injection yields the same S as .sp's power-wave excitation; for a time-invariant
 * circuit the conversion matrix is block-diagonal and the sideband-0 block reduces
 * exactly to the ordinary .sp S-matrix. */
static void
psp_sweep(CKTcircuit *ckt, PSSan *job)
{
    int    N = job->PSSopMsize, np = ckt->CKTportCount;
    int    M, Ksb, nsb, stepType = job->PACstepType, npts = job->PACpoints;
    int    i, j, s, p, numNames, error;
    double f0 = job->PSSopFreq, fstart = job->PACfStart, fstop = job->PACfStop;
    double freq, mult, linstep;
    struct pac_harm hd;
    double *Xr, *Xi;
    IFuid  freqUid, *outNames = NULL;
    runDesc *plot = NULL;
    CMat   *Amat = NULL, *Ainv = NULL, **Bmat = NULL;

    if (np < 1) {
        fprintf(stderr, "PSP: no RF ports (add `portnum`/`z0` to voltage sources) "
                        "-- sweep skipped\n");
        return;
    }
    if (f0 <= 0.0 || fstart <= 0.0 || fstop < fstart || npts < 1)
        return;

    /* every port's nodes + branch must live inside the conversion-matrix unknowns */
    for (p = 0; p < np; p++) {
        VSRCinstance *pr = (VSRCinstance *) ckt->CKTrfPorts[p];
        if (pr->VSRCbranch > N || pr->VSRCposNode > N || pr->VSRCnegNode > N) {
            fprintf(stderr, "PSP: port %d outside the PSS unknown set -- skipped\n",
                    pr->VSRCportNum);
            return;
        }
    }

    M = pac_choose_M(ckt, job);
    if (M < 1) { fprintf(stderr, "PSP: conversion matrix too large -- skipped\n"); return; }
    if (pac_extract_harmonics(ckt, job, M, &hd))
        return;

    Ksb = job->PACmaxSideband;
    if (Ksb < 0) Ksb = 0;
    if (Ksb > M) Ksb = M;
    nsb = 2 * Ksb + 1;

    /* output vectors: S_<dest>_<src> per sideband (dest = measured port, src =
     * excited).  Sideband 0 uses the plain name so it lines up with an ordinary
     * .sp run; conversion sidebands get an _usb<k>/_lsb<k> suffix. */
    numNames = nsb * np * np;
    outNames = TMALLOC(IFuid, numNames);
    {
        int idx = 0;
        for (s = 0; s < nsb; s++) {
            int k = s - Ksb;
            for (j = 1; j <= np; j++)
                for (i = 1; i <= np; i++) {
                    char nm[64];
                    if (k == 0)
                        snprintf(nm, sizeof(nm), "S_%d_%d", j, i);
                    else
                        snprintf(nm, sizeof(nm), "S_%d_%d_%csb%d",
                                 j, i, (k > 0) ? 'u' : 'l', abs(k));
                    SPfrontEnd->IFnewUid(ckt, &outNames[idx++], NULL, nm, UID_OTHER, NULL);
                }
        }
    }
    SPfrontEnd->IFnewUid(ckt, &freqUid, NULL, "frequency", UID_OTHER, NULL);
    error = SPfrontEnd->OUTpBeginPlot(ckt, ckt->CKTcurJob, "PSP Analysis",
                                      freqUid, IF_REAL, numNames, outNames,
                                      IF_COMPLEX, &plot);
    tfree(outNames);
    if (error) { pac_free_harmonics(&hd); return; }
    if (stepType != 0)
        SPfrontEnd->OUTattributes(plot, NULL, OUT_SCALE_LOG, NULL);

    Xr = TMALLOC(double, hd.Ntot);
    Xi = TMALLOC(double, hd.Ntot);
    Amat = newcmat(np, np, 0.0, 0.0);
    Bmat = TMALLOC(CMat *, nsb);
    for (s = 0; s < nsb; s++)
        Bmat[s] = newcmat(np, np, 0.0, 0.0);

    mult    = (stepType == 1) ? pow(10.0, 1.0 / npts) :
              (stepType == 2) ? pow(2.0,  1.0 / npts) : 0.0;
    linstep = (npts > 1) ? (fstop - fstart) / (npts - 1) : 0.0;

    fprintf(stderr, "PSP sweep: %s from %.6g to %.6g Hz around f0 = %.6g Hz; "
                    "%d port%s, %d sideband%s\n",
            (stepType == 1) ? "dec" : (stepType == 2) ? "oct" : "lin",
            fstart, fstop, f0, np, (np == 1) ? "" : "s", nsb, (nsb == 1) ? "" : "s");

    for (freq = fstart; freq <= fstop * (1.0 + 1e-9); ) {
        int ok = 1;
        /* excite each port -> one column of A (sb0) and of each B^(k) */
        for (i = 1; i <= np && ok; i++) {
            VSRCinstance *pi = (VSRCinstance *) ckt->CKTrfPorts[i - 1];
            if (psp_solve_port(&hd, f0, freq, pi->VSRCbranch, Xr, Xi)) {
                ok = 0;
                break;
            }
            for (j = 1; j <= np; j++) {
                VSRCinstance *pj = (VSRCinstance *) ckt->CKTrfPorts[j - 1];
                double ki = pj->VSRCki, z0 = pj->VSRCportZ0;
                int    pn = pj->VSRCposNode, nn = pj->VSRCnegNode, br = pj->VSRCbranch;
                for (s = 0; s < nsb; s++) {
                    int    k = s - Ksb;
                    size_t blk = (size_t)(k + M) * (size_t)N;
                    double Vr = 0.0, Vi = 0.0, Ir, Ii;
                    cplx   a, b;
                    if (pn > 0) { Vr += Xr[blk + (size_t)(pn - 1)]; Vi += Xi[blk + (size_t)(pn - 1)]; }
                    if (nn > 0) { Vr -= Xr[blk + (size_t)(nn - 1)]; Vi -= Xi[blk + (size_t)(nn - 1)]; }
                    Ir = -Xr[blk + (size_t)(br - 1)];
                    Ii = -Xi[blk + (size_t)(br - 1)];
                    b.re = ki * (Vr - z0 * Ir);
                    b.im = ki * (Vi - z0 * Ii);
                    setc(Bmat[s], j - 1, i - 1, b);
                    if (k == 0) {
                        a.re = ki * (Vr + z0 * Ir);
                        a.im = ki * (Vi + z0 * Ii);
                        setc(Amat, j - 1, i - 1, a);
                    }
                }
            }
        }
        if (ok && (Ainv = cinverse(Amat)) != NULL) {
            IFvalue freqData, valueData;
            IFcomplex *data = TMALLOC(IFcomplex, numNames);
            int idx = 0;
            freqData.rValue = freq;
            valueData.v.numValue = numNames;
            valueData.v.vec.cVec = data;
            for (s = 0; s < nsb; s++) {
                CMat *S = cmultiply(Bmat[s], Ainv);      /* S^(k) = B^(k) * A^-1 */
                for (j = 0; j < np; j++)
                    for (i = 0; i < np; i++) {
                        cplx sij = getcplx(S, j, i);
                        data[idx].real = sij.re;
                        data[idx].imag = sij.im;
                        idx++;
                    }
                freecmat(S);
            }
            SPfrontEnd->OUTpData(plot, &freqData, &valueData);
            FREE(data);
            freecmat(Ainv);
            Ainv = NULL;
        }
        if (stepType == 0) { if (npts <= 1) break; freq += linstep; }
        else               { freq *= mult; }
    }

    SPfrontEnd->OUTendPlot(plot);
    for (s = 0; s < nsb; s++)
        freecmat(Bmat[s]);
    FREE(Bmat);
    freecmat(Amat);
    FREE(Xr); FREE(Xi);
    pac_free_harmonics(&hd);
}
#endif


/* ===================== Enhancement-134: Harmonic Balance ===================== */
/*
 * Single-tone HB: solve the periodic steady state in the FREQUENCY domain. Each
 * node voltage is a truncated Fourier series V(t) = sum_{k=-K..K} V_k e^{jk w0 t};
 * the KCL residual at every node/harmonic
 *      F_k = I_R,k(V) + [dq/dt]_k - Is_k = 0
 * is solved by Newton, with the (2K+1)N conversion matrix (E-121) as the Jacobian.
 *  - I_R(v(t_s)) : nonlinear RESISTIVE current, from a DC-mode device load at each of
 *    P time samples (residual current = G*v - rhs).
 *  - [dq/dt]_k   : the REACTIVE current. dq/dt = C(v)*v' (chain rule), so its spectrum
 *    is the conversion matrix's reactive term (jm*w0*C_{k-m}) applied to V -- NONLINEAR
 *    charge is handled with NO per-device charge extraction, just the C(t) samples.
 *  - Is         : the independent-source excitation spectrum (loaded at v=0, t=t_s).
 * Reuses pac_build_matrix (Jacobian) and pss_csolve (dense complex Newton solve).
 */

/* Sample the device residual + Jacobian at prescribed node voltages vsamp[s*N+(i-1)]
 * (P samples). Fills hd with the G(t)/C(t) harmonics (h=0..2K) and returns the
 * resistive-current harmonics in IRr/IRi (length (2K+1)*N). Returns 0 on success. */
static int
hb_extract(CKTcircuit *ckt, const double *vsamp, int N, int P, int K,
           struct pac_harm *hd, double *IRr, double *IRi)
{
    int    H = 2 * K, i, r, c, e, h, nnz, s;
    int    *rr, *cc;
    double *Gt, *Ct, *IRt, *cw, *sw, *Gmr, *Gmi, *Cmr, *Cmi, *bsave;

    memset(hd, 0, sizeof(*hd));
    if (P <= 0 || N <= 0 || K < 1)
        return 1;
    bsave = TMALLOC(double, N);

    /* Matrix must be in complex mode so CKTacLoad stamps G(el[0]) + jC(el[1]) and
     * SMPfindElt reads both parts. Under KLU the live matrix is the CSC form, not
     * SPmatrix, so bind the device pointers to the complex CSC (mirrors
     * pac_extract_harmonics -- this is what makes HB solver-independent). */
#ifdef KLU
    if (ckt->CKTmatrix->CKTkluMODE) {
        if (!ckt->CKTmatrix->SMPkluMatrix->KLUmatrixIsComplex) {
            for (i = 0; i < DEVmaxnum; i++)
                if (DEVices[i] && DEVices[i]->DEVbindCSCComplex && ckt->CKThead[i])
                    DEVices[i]->DEVbindCSCComplex(ckt->CKThead[i], ckt);
            ckt->CKTmatrix->SMPkluMatrix->KLUmatrixIsComplex = KLUMatrixComplex;
        }
    } else
#endif
        spSetComplex(ckt->CKTmatrix->SPmatrix);

    /* establish structure at sample 0's bias */
    for (i = 1; i <= N; i++)
        ckt->CKTrhsOld[i] = vsamp[i - 1];
    ckt->CKTrhsOld[0] = 0.0;
    ckt->CKTmode = (ckt->CKTmode & MODEUIC) | MODEDCOP | MODEINITSMSIG;
    CKTload(ckt);
    ckt->CKTomega = 1.0;
    ckt->CKTmode = (ckt->CKTmode & MODEUIC) | MODEAC;
    CKTacLoad(ckt);

    nnz = 0;
    for (r = 1; r <= N; r++)
        for (c = 1; c <= N; c++)
            if (SMPfindElt(ckt->CKTmatrix, r, c, 0))
                nnz++;
    if (nnz <= 0)
        return 1;
    rr = TMALLOC(int, nnz);
    cc = TMALLOC(int, nnz);
    e = 0;
    for (r = 1; r <= N; r++)
        for (c = 1; c <= N; c++)
            if (SMPfindElt(ckt->CKTmatrix, r, c, 0)) { rr[e] = r; cc[e] = c; e++; }

    Gt  = TMALLOC(double, (size_t)nnz * (size_t)P);
    Ct  = TMALLOC(double, (size_t)nnz * (size_t)P);
    IRt = TMALLOC(double, (size_t)N * (size_t)P);

    for (s = 0; s < P; s++) {
        double *Gv;
        /* DC-mode load at v(t_s): matrix real = G, rhs = resistive companion */
        for (i = 1; i <= N; i++)
            ckt->CKTrhsOld[i] = vsamp[(size_t)s * (size_t)N + (size_t)(i - 1)];
        ckt->CKTrhsOld[0] = 0.0;
        for (i = 0; i <= N; i++)
            ckt->CKTrhs[i] = 0.0;
        /* (a) Settle the nonlinear device state at the FIXED node voltages v(t_s).
         * In MODEINITFLOAT a junction device (diode/BJT/MOS) reads the node voltage
         * but LIMITS the junction step against its stored internal voltage; a single
         * load leaves it pinned at a stale bias (and MODEINITSMSIG alone reads the
         * stored op-point, ignoring the node voltage -- the bug that made real diodes
         * look linear). So load repeatedly: each pass walks the internal voltage
         * toward the fixed node voltages until the limiter is a no-op. Behavioural/
         * OSDI devices with no limiting settle on the first pass. */
        ckt->CKTmode = (ckt->CKTmode & MODEUIC) | MODEDCOP | MODEINITFLOAT;
        {
            int inner;
            for (inner = 0; inner < 100; inner++) {
                double bnorm = 0.0, dnorm = 0.0;
                for (i = 0; i <= N; i++)
                    ckt->CKTrhs[i] = 0.0;
                CKTload(ckt);                /* reads CKTrhsOld (fixed v), evolves state0 */
                for (i = 1; i <= N; i++) {
                    double db = ckt->CKTrhs[i] - bsave[i - 1];
                    dnorm += db * db;
                    bnorm += ckt->CKTrhs[i] * ckt->CKTrhs[i];
                    bsave[i - 1] = ckt->CKTrhs[i];
                }
                if (inner > 0 && sqrt(dnorm) <= 1e-12 * (sqrt(bnorm) + 1e-30))
                    break;
            }
        }
        /* The settled MODEINITFLOAT load left the DC companion b = G*v - i(v) in
         * bsave; it gives the ACTUAL resistive current i(v) = G*v - b for EVERY
         * device (behavioural, OSDI, junction), not the tangent G*v.
         * (b) A MODEINITSMSIG load then reads the now-settled junction state to build
         * the small-signal linearization the AC (C) load needs. Its RHS is NOT the DC
         * companion, so we do NOT overwrite bsave; but the small-signal G it sets up
         * is the same di/dv at the settled bias, so I_R = G*v - b_float = i(v). */
        for (i = 0; i <= N; i++)
            ckt->CKTrhs[i] = 0.0;
        ckt->CKTmode = (ckt->CKTmode & MODEUIC) | MODEDCOP | MODEINITSMSIG;
        CKTload(ckt);
        ckt->CKTomega = 1.0;
        ckt->CKTmode = (ckt->CKTmode & MODEUIC) | MODEAC;
        CKTacLoad(ckt);   /* clears + stamps G (real) and C (imag) at the settled bias */

        /* resistive current I_R = G*v - b, using the clean G from acLoad */
        Gv = IRt + (size_t)s * (size_t)N;   /* reuse row s as scratch, then subtract */
        for (i = 0; i < N; i++)
            Gv[i] = 0.0;
        for (e = 0; e < nnz; e++) {
            double *el = (double *) SMPfindElt(ckt->CKTmatrix, rr[e], cc[e], 0);
            double g = el ? el[0] : 0.0;
            Gt[(size_t)e * (size_t)P + (size_t)s] = g;
            Ct[(size_t)e * (size_t)P + (size_t)s] = el ? el[1] : 0.0;
            Gv[rr[e] - 1] += g * vsamp[(size_t)s * (size_t)N + (size_t)(cc[e] - 1)];
        }
        for (i = 1; i <= N; i++)
            Gv[i - 1] -= bsave[i - 1];      /* I_R = G*v - b = i(v) */
    }

    /* DFT G(t), C(t) -> harmonics; I_R(t) -> IRr/IRi (harmonics -K..K packed 0..2K) */
    cw = TMALLOC(double, (size_t)(H + 1) * (size_t)P);
    sw = TMALLOC(double, (size_t)(H + 1) * (size_t)P);
    for (h = 0; h <= H; h++)
        for (s = 0; s < P; s++) {
            double ang = 2.0 * M_PI * (double)h * (double)s / (double)P;
            cw[(size_t)h * (size_t)P + (size_t)s] = cos(ang);
            sw[(size_t)h * (size_t)P + (size_t)s] = sin(ang);
        }
    Gmr = TMALLOC(double, (size_t)nnz * (size_t)(H + 1));
    Gmi = TMALLOC(double, (size_t)nnz * (size_t)(H + 1));
    Cmr = TMALLOC(double, (size_t)nnz * (size_t)(H + 1));
    Cmi = TMALLOC(double, (size_t)nnz * (size_t)(H + 1));
    for (e = 0; e < nnz; e++)
        for (h = 0; h <= H; h++) {
            double gr = 0, gi = 0, cr = 0, ci = 0;
            for (s = 0; s < P; s++) {
                double cs = cw[(size_t)h * (size_t)P + (size_t)s];
                double sn = sw[(size_t)h * (size_t)P + (size_t)s];
                double gv = Gt[(size_t)e * (size_t)P + (size_t)s];
                double cv = Ct[(size_t)e * (size_t)P + (size_t)s];
                gr += gv * cs;  gi -= gv * sn;
                cr += cv * cs;  ci -= cv * sn;
            }
            Gmr[(size_t)e * (size_t)(H + 1) + (size_t)h] = gr / (double)P;
            Gmi[(size_t)e * (size_t)(H + 1) + (size_t)h] = gi / (double)P;
            Cmr[(size_t)e * (size_t)(H + 1) + (size_t)h] = cr / (double)P;
            Cmi[(size_t)e * (size_t)(H + 1) + (size_t)h] = ci / (double)P;
        }
    /* I_R harmonics: full k=-K..K (row (k+K)*N + node) */
    for (r = 0; r < N; r++)
        for (h = -K; h <= K; h++) {
            double xr = 0, xi = 0;
            int hh = h < 0 ? -h : h;
            for (s = 0; s < P; s++) {
                double cs = cw[(size_t)hh * (size_t)P + (size_t)s];
                double sn = sw[(size_t)hh * (size_t)P + (size_t)s];
                double x  = IRt[(size_t)s * (size_t)N + (size_t)r];
                if (h >= 0) { xr += x * cs; xi -= x * sn; }
                else        { xr += x * cs; xi += x * sn; }   /* conj for -h */
            }
            IRr[(size_t)(h + K) * (size_t)N + (size_t)r] = xr / (double)P;
            IRi[(size_t)(h + K) * (size_t)N + (size_t)r] = xi / (double)P;
        }

    FREE(Gt); FREE(Ct); FREE(IRt); FREE(cw); FREE(sw); FREE(bsave);
    hd->N = N; hd->M = K; hd->H = H; hd->nnz = nnz; hd->Ntot = (2*K + 1) * N;
    hd->rr = rr; hd->cc = cc;
    hd->Gmr = Gmr; hd->Gmi = Gmi; hd->Cmr = Cmr; hd->Cmi = Cmi;
    hd->B0r = NULL; hd->B0i = NULL; hd->has_src = 0;
    return 0;
}

int
HBanalyze(CKTcircuit *ckt, double f0, int K, int Pin, int maxiter, double tol, int verbose, struct hbspectrum *out)
{
    int    N = SMPmatSize(ckt->CKTmatrix);
    int    P = Pin > 0 ? Pin : ((8 * K < 32) ? 32 : 8 * K);
    int    Ntot = (2 * K + 1) * N;
    int    iter, s, i, k, rc = 0;
    double T = 1.0 / f0, w0 = 2.0 * M_PI * f0;
    double *Vr, *Vi, *vsamp, *IRr, *IRi, *Isr, *Isi, *Fr, *Fi, *Jr, *Ji, *Kr, *Ki;
    struct pac_harm hd;

    if (N <= 0 || K < 1) { fprintf(stderr, "HB: bad size.\n"); return E_PARMVAL; }
    if ((2 * K + 1) * N > 900) {
        fprintf(stderr, "HB: system %dx%d too large for the dense solver.\n", Ntot, Ntot);
        return E_PARMVAL;
    }

#ifdef HAS_PROGREP
    SetAnalyse("HB", 0);
#endif

    Vr = TMALLOC(double, Ntot); Vi = TMALLOC(double, Ntot);
    IRr = TMALLOC(double, Ntot); IRi = TMALLOC(double, Ntot);
    Isr = TMALLOC(double, Ntot); Isi = TMALLOC(double, Ntot);
    Fr = TMALLOC(double, Ntot); Fi = TMALLOC(double, Ntot);
    Kr = TMALLOC(double, Ntot); Ki = TMALLOC(double, Ntot);
    Jr = TMALLOC(double, (size_t)Ntot * (size_t)Ntot); Ji = TMALLOC(double, (size_t)Ntot * (size_t)Ntot);
    vsamp = TMALLOC(double, (size_t)N * (size_t)P);

    /* --- source spectrum Is_k: load at v=0, sources evaluated at t_s --- */
    {
        double *ist = TMALLOC(double, (size_t)N * (size_t)P);
        for (s = 0; s < P; s++) {
            for (i = 0; i <= N; i++) { ckt->CKTrhsOld[i] = 0.0; ckt->CKTrhs[i] = 0.0; }
            ckt->CKTtime = (double)s * T / (double)P;
            ckt->CKTmode = (ckt->CKTmode & MODEUIC) | MODETRAN | MODEINITTRAN;
            CKTload(ckt);
            for (i = 1; i <= N; i++)
                ist[(size_t)s * (size_t)N + (size_t)(i - 1)] = ckt->CKTrhs[i];
        }
        for (i = 0; i < N; i++)
            for (k = -K; k <= K; k++) {
                double xr = 0, xi = 0; int hh = k < 0 ? -k : k;
                for (s = 0; s < P; s++) {
                    double ang = 2.0 * M_PI * hh * s / (double)P;
                    double x = ist[(size_t)s * (size_t)N + (size_t)i];
                    xr += x * cos(ang);
                    xi += (k >= 0 ? -1.0 : 1.0) * x * sin(ang);
                }
                Isr[(size_t)(k + K) * (size_t)N + (size_t)i] = xr / P;
                Isi[(size_t)(k + K) * (size_t)N + (size_t)i] = xi / P;
            }
        FREE(ist);
    }

    /* --- source-stepping continuation ------------------------------------
     * Solve HB with every independent source scaled by a homotopy factor
     * lambda: 0 -> 1, each level warm-started from the last converged point.
     * Adaptive with backtracking: the first level is full strength
     * (dlambda = 1), so an easy circuit converges at lambda = 1 on the first
     * try -- bit-identical to the plain direct solve. When a level fails
     * (Newton runs out, the residual goes non-finite, or the Jacobian is
     * singular) the step is halved and retried from the last converged V; when
     * a level converges the step grows. This carries a strongly-driven circuit
     * (nonlinearity comparable to the linear term -- e.g. a PA near
     * compression) to steady state where a cold full-strength Newton diverges.
     * All independent sources (bias and drive) ramp together -- classic source
     * stepping; to sweep drive at fixed bias, step the drive with `alter`. */
    {
    double lambda = 0.0, dlambda = 1.0, fnorm = 0.0;
    double *Vsr = TMALLOC(double, Ntot);
    double *Vsi = TMALLOC(double, Ntot);
    int nlevels = 0, nnewton = 0, hard_err = 0;
    memcpy(Vsr, Vr, (size_t)Ntot * sizeof(double)); /* last-good = cold V=0 (lambda=0 solution) */
    memcpy(Vsi, Vi, (size_t)Ntot * sizeof(double));

    for (;;) {
        double target = lambda + dlambda;
        int conv = 0;
        if (target > 1.0)
            target = 1.0;

        for (iter = 0; iter < maxiter; iter++) {
            /* v(t_s) = Re sum_k V_k e^{j k w0 t_s} */
            for (s = 0; s < P; s++)
                for (i = 0; i < N; i++) {
                    double v = 0.0;
                    for (k = -K; k <= K; k++) {
                        double ang = 2.0 * M_PI * k * s / (double)P;
                        v += Vr[(size_t)(k + K) * (size_t)N + (size_t)i] * cos(ang)
                           - Vi[(size_t)(k + K) * (size_t)N + (size_t)i] * sin(ang);
                    }
                    vsamp[(size_t)s * (size_t)N + (size_t)i] = v;
                }

            if (hb_extract(ckt, vsamp, N, P, K, &hd, IRr, IRi)) {
                fprintf(stderr, "HB: device extraction failed.\n");
                rc = E_PARMVAL; hard_err = 1; break;
            }

            /* full Jacobian J = G + jwC conversion matrix */
            pac_build_matrix(&hd, f0, 0.0, Jr, Ji, 0);

            /* reactive current I_C = (J - Jg)*V where Jg is the resistive
             * (G-only) conversion matrix -- i.e. the jwC part of J on V. */
            {
                struct pac_harm hg = hd;
                double *Jgr = TMALLOC(double, (size_t)Ntot * (size_t)Ntot);
                double *Jgi = TMALLOC(double, (size_t)Ntot * (size_t)Ntot);
                hg.Cmr = TMALLOC(double, (size_t)hd.nnz * (size_t)(hd.H + 1));  /* zero C -> resistive only */
                hg.Cmi = TMALLOC(double, (size_t)hd.nnz * (size_t)(hd.H + 1));
                pac_build_matrix(&hg, f0, 0.0, Jgr, Jgi, 0);
                FREE(hg.Cmr); FREE(hg.Cmi);
                for (i = 0; i < Ntot; i++) {
                    double cr = 0, ci = 0;
                    for (k = 0; k < Ntot; k++) {
                        double ar = Jr[(size_t)i * (size_t)Ntot + (size_t)k] - Jgr[(size_t)i * (size_t)Ntot + (size_t)k];
                        double ai = Ji[(size_t)i * (size_t)Ntot + (size_t)k] - Jgi[(size_t)i * (size_t)Ntot + (size_t)k];
                        cr += ar * Vr[k] - ai * Vi[k];
                        ci += ar * Vi[k] + ai * Vr[k];
                    }
                    Kr[i] = cr; Ki[i] = ci;     /* Kr/Ki = reactive current I_C */
                }
                FREE(Jgr); FREE(Jgi);
            }
            pac_free_harmonics(&hd);            /* hd not needed past this point */

            /* residual F = I_R + I_C - lambda*Is */
            for (i = 0; i < Ntot; i++) {
                Fr[i] = IRr[i] + Kr[i] - target * Isr[i];
                Fi[i] = IRi[i] + Ki[i] - target * Isi[i];
            }
            /* Enhancement-178: the settle-mode rhs folded into I_R already
             * carries the DC (operating-point) source values, so -lambda*Is
             * on its own would subtract the DC sources a SECOND time: the
             * converged spectrum answered a doubled DC drive (every DC bias
             * voltage came out exactly 2x, silently corrupting all
             * bias-dependent noise and conversion downstream -- caught by the
             * E-178 cyclostationary referee). Adding the UNSCALED DC source
             * block back makes the net DC drive exactly -lambda*Is_DC.
             * (Assumes the DCOP source value equals the tone-basis DC
             * average, true for DC/SIN sources -- the HB use cases.) */
            for (i = 0; i < N; i++) {
                Fr[(size_t)K * (size_t)N + (size_t)i] += Isr[(size_t)K * (size_t)N + (size_t)i];
                Fi[(size_t)K * (size_t)N + (size_t)i] += Isi[(size_t)K * (size_t)N + (size_t)i];
            }
            fnorm = 0.0;
            for (i = 0; i < Ntot; i++)
                fnorm += Fr[i] * Fr[i] + Fi[i] * Fi[i];
            fnorm = sqrt(fnorm);
            nnewton++;
            if (verbose)
                fprintf(stderr, "HB  lambda=%.4f  iter %2d: |F| = %.6e\n", target, iter, fnorm);

            if (isnan(fnorm) || fnorm > 1e300)
                break;                          /* diverged -> level fails */

            /* Newton step: J * dV = -F */
            for (i = 0; i < Ntot; i++) { Fr[i] = -Fr[i]; Fi[i] = -Fi[i]; }
            if (pss_csolve(Ntot, Jr, Ji, Fr, Fi))   /* singular -> level fails */
                break;
            for (i = 0; i < Ntot; i++) { Vr[i] += Fr[i]; Vi[i] += Fi[i]; }

            if (fnorm < tol) { conv = 1; break; }
        }

        if (hard_err)
            break;

        if (conv) {
            lambda = target;
            nlevels++;
            memcpy(Vsr, Vr, (size_t)Ntot * sizeof(double)); /* checkpoint this level */
            memcpy(Vsi, Vi, (size_t)Ntot * sizeof(double));
            if (lambda >= 1.0 - 1e-9)
                break;                          /* reached full strength */
            dlambda *= 1.7;                     /* grow the step while it's easy */
            if (lambda + dlambda > 1.0)
                dlambda = 1.0 - lambda;
        } else {
            memcpy(Vr, Vsr, (size_t)Ntot * sizeof(double)); /* restore last good */
            memcpy(Vi, Vsi, (size_t)Ntot * sizeof(double));
            dlambda *= 0.5;
            if (dlambda < 1e-5) {
                fprintf(stderr, "HB: source stepping stalled at lambda=%.4g "
                                "(|F|=%.3e); the circuit may be singular or have "
                                "no periodic steady state at this drive.\n",
                        lambda, fnorm);
                rc = E_ITERLIM;
                break;
            }
        }
    }

    if (rc == OK)
        fprintf(stdout, "HB: converged in %d iterations, %d continuation step%s "
                        "(|F| = %.3e).\n",
                nnewton, nlevels, nlevels == 1 ? "" : "s", fnorm);
    FREE(Vsr);
    FREE(Vsi);
    }

    /* --- output: labelled spectrum table, magnitude per node per harmonic --- */
    {
        int numNames, error;
        IFuid *nameList = NULL;
        error = CKTnames(ckt, &numNames, &nameList);
        fprintf(stdout, "\nHB: harmonic-balance spectrum (f0 = %g Hz, %d harmonics)\n"
                        "  node      harmonic   frequency [Hz]        |V|            phase [deg]\n",
                f0, K);
        for (i = 0; i < N; i++) {
            const char *nm = (!error && i < numNames) ? (const char *) nameList[i] : "?";
            for (k = 0; k <= K; k++) {
                double sc = (k == 0) ? 1.0 : 2.0;   /* single-sided amplitude */
                double vr = sc * Vr[(size_t)(k + K) * (size_t)N + (size_t)i];
                double vi = sc * Vi[(size_t)(k + K) * (size_t)N + (size_t)i];
                fprintf(stdout, "  %-8s  %6d   %16.6e   %14.6e   %10.3f\n",
                        nm, k, k * f0, hypot(vr, vi),
                        (k == 0) ? 0.0 : atan2(vi, vr) * 180.0 / M_PI);
            }
        }
        if (nameList) tfree(nameList);
    }
    (void) w0;

    /* Enhancement-209: hand the converged two-sided spectrum to the frontend so
       `com_hb` can publish it as nutmeg vectors (hbfrequency + one complex vector
       per node). Ownership of Vr/Vi passes to the caller; NULL them here so the
       cleanup below does not free them. */
    if (out && rc == OK) {
        out->N = N; out->K = K; out->f0 = f0;
        out->Vr = Vr; out->Vi = Vi;
        Vr = NULL; Vi = NULL;
    }

    FREE(Vr); FREE(Vi); FREE(IRr); FREE(IRi); FREE(Isr); FREE(Isi);
    FREE(Fr); FREE(Fi); FREE(Jr); FREE(Ji); FREE(Kr); FREE(Ki); FREE(vsamp);
    return rc;
}


/* ======================================================================
 * Enhancement-136: frequency-domain two-tone Harmonic Balance -- the TRUE
 * (incommensurate-capable) quasi-periodic steady state, `qpss ... hb`.
 *
 * Each node voltage is a 2-D Fourier series
 *   v(t) = sum_{k1=-K1..K1, k2=-K2..K2} V_{k1,k2} e^{j(k1 w1 + k2 w2) t}.
 * Devices are sampled on a 2-D PHASE grid (theta1,theta2) -- set the node
 * voltages, load, read G/C -- which is time-independent, so INCOMMENSURATE
 * tones just work (no common period); a 2-D DFT gives G_{d1,d2}, C_{d1,d2}.
 * The source spectrum is captured by an almost-periodic Fourier transform
 * (APFT): sample the v=0 source RHS at Nh real times and invert the
 * Vandermonde. The 2-D conversion matrix H_{(n),(m)} = G_{n-m}+j*w_m*C_{n-m}
 * is the direct analogue of pac_build_matrix; the Newton (with E-135 source
 * stepping) reuses pss_csolve. The converged operating point + conversion
 * data are retained in qpss_hb_saved for `qpac` (Enhancement-137).
 * ====================================================================== */

struct qp_harm {
    int  N, K1, K2, Nh, Ntot, nnz;
    int *rr, *cc;              /* [nnz] 1-based Jacobian nonzero row/col */
    int *h1, *h2;              /* [Nh] (k1,k2) of each harmonic index */
    int  D2c, Dsz;             /* diff-spectrum stride (4K2+1), size (4K1+1)(4K2+1) */
    double *Gmr, *Gmi, *Cmr, *Cmi;   /* [nnz*Dsz] 2-D difference spectra */
    double f1, f2;
    double *Vr, *Vi;           /* [Ntot] retained operating point (for qpac); else NULL */
    double *B0r, *B0i;         /* [N] AC-source stimulus for qpac (E-137); else NULL */
    int     has_src;           /* 1 if a netlist AC source stamped B0 */
    int     P1, P2;            /* phase-grid used for extraction (for cyclostationary qpnoise, E-139) */
};

static struct qp_harm *qpss_hb_saved = NULL;   /* retained QPSS op-point for qpac (E-137) */

static int qp_didx(const struct qp_harm *h, int d1, int d2)
{ return (d1 + 2*h->K1) * h->D2c + (d2 + 2*h->K2); }

/* free the arrays a qp_harm owns (not the struct itself) */
static void qp_free(struct qp_harm *hd)
{
    if (!hd) return;
    FREE(hd->rr); FREE(hd->cc); FREE(hd->h1); FREE(hd->h2);
    FREE(hd->Gmr); FREE(hd->Gmi); FREE(hd->Cmr); FREE(hd->Cmi);
    FREE(hd->Vr); FREE(hd->Vi); FREE(hd->B0r); FREE(hd->B0i);
}

/* assemble the dense Ntot x Ntot 2-D conversion matrix at input freq f_in.
 * smallsig selects the reactive frequency multiplier exactly as in
 * pac_build_matrix (RF-audit fix): 1 = ROW frequency (LPTV small-signal:
 * qpac/qpnoise/qpxf), 0 = COLUMN frequency (QPSS-HB residual/Jacobian,
 * chain rule). */
static void qp_build_matrix(struct qp_harm *hd, double f_in, double *Ar, double *Ai,
                            int smallsig)
{
    int Nh = hd->Nh, N = hd->N, Ntot = hd->Ntot, nnz = hd->nnz, ni, mi, e;
    memset(Ar, 0, (size_t)Ntot * (size_t)Ntot * sizeof(double));
    memset(Ai, 0, (size_t)Ntot * (size_t)Ntot * sizeof(double));
    for (ni = 0; ni < Nh; ni++)
        for (mi = 0; mi < Nh; mi++) {
            int d1 = hd->h1[ni] - hd->h1[mi], d2 = hd->h2[ni] - hd->h2[mi];
            int fi = smallsig ? ni : mi;
            double omega = 2.0 * M_PI * (f_in + hd->h1[fi]*hd->f1 + hd->h2[fi]*hd->f2);
            int di = qp_didx(hd, d1, d2);
            for (e = 0; e < nnz; e++) {
                size_t hi = (size_t)e * (size_t)hd->Dsz + (size_t)di;
                double gr = hd->Gmr[hi], gi = hd->Gmi[hi];
                double cr = hd->Cmr[hi], ci = hd->Cmi[hi];
                size_t row = (size_t)ni * (size_t)N + (size_t)(hd->rr[e] - 1);
                size_t col = (size_t)mi * (size_t)N + (size_t)(hd->cc[e] - 1);
                Ar[row * (size_t)Ntot + col] += gr - omega * ci;
                Ai[row * (size_t)Ntot + col] += gi + omega * cr;
            }
        }
}

/* Sample the device Jacobian G(theta1,theta2), C(theta1,theta2) and resistive
 * current I_R on the P1xP2 phase grid at prescribed node voltages, and 2-D DFT
 * to the difference spectra + I_R harmonics. Fills hd; returns 0/1. Mirrors
 * hb_extract (E-134) but on the 2-D phase grid. */
static int
qp_extract(CKTcircuit *ckt, const double *vsamp, int N, int P1, int P2,
           int K1, int K2, struct qp_harm *hd, double *IRr, double *IRi)
{
    int P = P1 * P2, i, r, c, e, s, s1, s2, nnz, d1, d2, hi;
    int *rr, *cc;
    double *Gt, *Ct, *IRt, *bsave;

    memset(hd, 0, sizeof(*hd));
    if (P <= 0 || N <= 0 || K1 < 1 || K2 < 1) return 1;
    bsave = TMALLOC(double, N);

#ifdef KLU
    if (ckt->CKTmatrix->CKTkluMODE) {
        if (!ckt->CKTmatrix->SMPkluMatrix->KLUmatrixIsComplex) {
            for (i = 0; i < DEVmaxnum; i++)
                if (DEVices[i] && DEVices[i]->DEVbindCSCComplex && ckt->CKThead[i])
                    DEVices[i]->DEVbindCSCComplex(ckt->CKThead[i], ckt);
            ckt->CKTmatrix->SMPkluMatrix->KLUmatrixIsComplex = KLUMatrixComplex;
        }
    } else
#endif
        spSetComplex(ckt->CKTmatrix->SPmatrix);

    /* establish structure at sample 0's bias */
    for (i = 1; i <= N; i++) ckt->CKTrhsOld[i] = vsamp[i - 1];
    ckt->CKTrhsOld[0] = 0.0;
    ckt->CKTmode = (ckt->CKTmode & MODEUIC) | MODEDCOP | MODEINITSMSIG;
    CKTload(ckt);
    ckt->CKTomega = 1.0;
    ckt->CKTmode = (ckt->CKTmode & MODEUIC) | MODEAC;
    CKTacLoad(ckt);

    nnz = 0;
    for (r = 1; r <= N; r++) for (c = 1; c <= N; c++)
        if (SMPfindElt(ckt->CKTmatrix, r, c, 0)) nnz++;
    if (nnz <= 0) { FREE(bsave); return 1; }
    rr = TMALLOC(int, nnz); cc = TMALLOC(int, nnz);
    e = 0;
    for (r = 1; r <= N; r++) for (c = 1; c <= N; c++)
        if (SMPfindElt(ckt->CKTmatrix, r, c, 0)) { rr[e] = r; cc[e] = c; e++; }

    Gt  = TMALLOC(double, (size_t)nnz * (size_t)P);
    Ct  = TMALLOC(double, (size_t)nnz * (size_t)P);
    IRt = TMALLOC(double, (size_t)N * (size_t)P);

    for (s = 0; s < P; s++) {
        double *Gv;
        for (i = 1; i <= N; i++)
            ckt->CKTrhsOld[i] = vsamp[(size_t)s * (size_t)N + (size_t)(i - 1)];
        ckt->CKTrhsOld[0] = 0.0;
        for (i = 0; i <= N; i++) ckt->CKTrhs[i] = 0.0;
        /* settle limited junctions (E-134): repeated MODEINITFLOAT loads */
        ckt->CKTmode = (ckt->CKTmode & MODEUIC) | MODEDCOP | MODEINITFLOAT;
        { int inner;
          for (inner = 0; inner < 100; inner++) {
            double bnorm = 0.0, dnorm = 0.0;
            for (i = 0; i <= N; i++) ckt->CKTrhs[i] = 0.0;
            CKTload(ckt);
            for (i = 1; i <= N; i++) {
                double db = ckt->CKTrhs[i] - bsave[i - 1];
                dnorm += db * db; bnorm += ckt->CKTrhs[i] * ckt->CKTrhs[i];
                bsave[i - 1] = ckt->CKTrhs[i];
            }
            if (inner > 0 && sqrt(dnorm) <= 1e-12 * (sqrt(bnorm) + 1e-30)) break;
          } }
        for (i = 0; i <= N; i++) ckt->CKTrhs[i] = 0.0;
        ckt->CKTmode = (ckt->CKTmode & MODEUIC) | MODEDCOP | MODEINITSMSIG;
        CKTload(ckt);
        ckt->CKTomega = 1.0;
        ckt->CKTmode = (ckt->CKTmode & MODEUIC) | MODEAC;
        CKTacLoad(ckt);
        Gv = IRt + (size_t)s * (size_t)N;
        for (i = 0; i < N; i++) Gv[i] = 0.0;
        for (e = 0; e < nnz; e++) {
            double *el = (double *) SMPfindElt(ckt->CKTmatrix, rr[e], cc[e], 0);
            double g = el ? el[0] : 0.0;
            Gt[(size_t)e * (size_t)P + (size_t)s] = g;
            Ct[(size_t)e * (size_t)P + (size_t)s] = el ? el[1] : 0.0;
            Gv[rr[e] - 1] += g * vsamp[(size_t)s * (size_t)N + (size_t)(cc[e] - 1)];
        }
        for (i = 1; i <= N; i++) Gv[i - 1] -= bsave[i - 1];   /* I_R = G*v - b */
    }

    hd->N = N; hd->K1 = K1; hd->K2 = K2; hd->Nh = (2*K1+1) * (2*K2+1);
    hd->Ntot = hd->Nh * N; hd->nnz = nnz; hd->rr = rr; hd->cc = cc;
    hd->D2c = 4*K2 + 1; hd->Dsz = (4*K1+1) * (4*K2+1);
    hd->P1 = P1; hd->P2 = P2;
    hd->h1 = TMALLOC(int, hd->Nh); hd->h2 = TMALLOC(int, hd->Nh);
    { int k1, k2; hi = 0;
      for (k1 = -K1; k1 <= K1; k1++) for (k2 = -K2; k2 <= K2; k2++) {
          hd->h1[hi] = k1; hd->h2[hi] = k2; hi++; } }

    hd->Gmr = TMALLOC(double, (size_t)nnz * (size_t)hd->Dsz);
    hd->Gmi = TMALLOC(double, (size_t)nnz * (size_t)hd->Dsz);
    hd->Cmr = TMALLOC(double, (size_t)nnz * (size_t)hd->Dsz);
    hd->Cmi = TMALLOC(double, (size_t)nnz * (size_t)hd->Dsz);
    /* 2-D DFT of G(theta1,theta2), C(theta1,theta2) -> difference spectra */
    for (e = 0; e < nnz; e++)
        for (d1 = -2*K1; d1 <= 2*K1; d1++)
            for (d2 = -2*K2; d2 <= 2*K2; d2++) {
                double gr = 0, gi = 0, cr = 0, ci = 0;
                for (s1 = 0; s1 < P1; s1++) for (s2 = 0; s2 < P2; s2++) {
                    int ss = s1 * P2 + s2;
                    double ang = 2.0*M_PI*((double)d1*s1/P1 + (double)d2*s2/P2);
                    double cs = cos(ang), sn = sin(ang);
                    double gv = Gt[(size_t)e * (size_t)P + (size_t)ss];
                    double cv = Ct[(size_t)e * (size_t)P + (size_t)ss];
                    gr += gv*cs; gi -= gv*sn; cr += cv*cs; ci -= cv*sn;
                }
                { size_t idx = (size_t)e * (size_t)hd->Dsz + (size_t)qp_didx(hd, d1, d2);
                  hd->Gmr[idx] = gr/P; hd->Gmi[idx] = gi/P;
                  hd->Cmr[idx] = cr/P; hd->Cmi[idx] = ci/P; }
            }
    /* I_R harmonics over the full Nh set (row hi*N + node) */
    for (r = 0; r < N; r++)
        for (hi = 0; hi < hd->Nh; hi++) {
            double xr = 0, xi = 0; int k1 = hd->h1[hi], k2 = hd->h2[hi];
            for (s1 = 0; s1 < P1; s1++) for (s2 = 0; s2 < P2; s2++) {
                int ss = s1 * P2 + s2;
                double ang = 2.0*M_PI*((double)k1*s1/P1 + (double)k2*s2/P2);
                double x = IRt[(size_t)ss * (size_t)N + (size_t)r];
                xr += x*cos(ang); xi -= x*sin(ang);
            }
            IRr[(size_t)hi * (size_t)N + (size_t)r] = xr/P;
            IRi[(size_t)hi * (size_t)N + (size_t)r] = xi/P;
        }

    FREE(Gt); FREE(Ct); FREE(IRt); FREE(bsave);
    return 0;
}

/* reconstruct v(theta1,theta2) samples [P1*P2 * N] from the 2-D spectrum V */
static void
qp_synth(const double *Vr, const double *Vi, int N, int K1, int K2,
         int P1, int P2, double *vsamp)
{
    int s1, s2, i, k1, k2, hi;
    for (s1 = 0; s1 < P1; s1++)
        for (s2 = 0; s2 < P2; s2++) {
            int s = s1 * P2 + s2;
            for (i = 0; i < N; i++) {
                double v = 0.0;
                hi = 0;
                for (k1 = -K1; k1 <= K1; k1++)
                    for (k2 = -K2; k2 <= K2; k2++) {
                        double ang = 2.0*M_PI*((double)k1*s1/P1 + (double)k2*s2/P2);
                        v += Vr[(size_t)hi*(size_t)N + (size_t)i] * cos(ang)
                           - Vi[(size_t)hi*(size_t)N + (size_t)i] * sin(ang);
                        hi++;
                    }
                vsamp[(size_t)s * (size_t)N + (size_t)i] = v;
            }
        }
}

int
QPSShb(CKTcircuit *ckt, double f1, double f2, int K1, int K2, int P1, int P2,
       int maxiter, double tol, int verbose)
{
    int N = SMPmatSize(ckt->CKTmatrix);
    int Nh = (2*K1+1) * (2*K2+1);
    int Ntot = Nh * N, P, i, hi, rc = OK;
    double *Vr, *Vi, *vsamp, *IRr, *IRi, *Isr, *Isi, *Fr, *Fi, *Jr, *Ji, *Kr, *Ki;
    struct qp_harm hd;

#ifdef HAS_PROGREP
    SetAnalyse("Two-tone HB", 0);
#endif

    if (N <= 0 || K1 < 1 || K2 < 1) { fprintf(stderr, "QPSS-HB: bad size.\n"); return E_PARMVAL; }
    if (P1 <= 0) P1 = 4*K1 + 2;
    if (P2 <= 0) P2 = 4*K2 + 2;
    P = P1 * P2;
    if (Ntot > 1600) {
        fprintf(stderr, "QPSS-HB: system %d too large for the dense solver "
                        "(reduce K1/K2).\n", Ntot);
        return E_PARMVAL;
    }

    /* APFT/conversion need all harmonic frequencies k1*f1+k2*f2 distinct */
    { int a, b, k1, k2, hh = 0; double fs = f1 + f2;
      int *t1 = TMALLOC(int, Nh), *t2 = TMALLOC(int, Nh);
      for (k1 = -K1; k1 <= K1; k1++) for (k2 = -K2; k2 <= K2; k2++) { t1[hh]=k1; t2[hh]=k2; hh++; }
      for (a = 0; a < Nh; a++) for (b = a+1; b < Nh; b++)
        if (fabs((t1[a]-t1[b])*f1 + (t2[a]-t2[b])*f2) < 1e-9 * fs) {
            fprintf(stderr, "QPSS-HB: tones f1=%g f2=%g are too commensurate at order "
                            "K1=%d K2=%d -- harmonics (%d,%d) and (%d,%d) alias to the same "
                            "frequency. Reduce the order or use the transient qpss.\n",
                    f1, f2, K1, K2, t1[a], t2[a], t1[b], t2[b]);
            FREE(t1); FREE(t2); return E_PARMVAL;
        }
      FREE(t1); FREE(t2);
    }

    Vr = TMALLOC(double, Ntot); Vi = TMALLOC(double, Ntot);
    IRr = TMALLOC(double, Ntot); IRi = TMALLOC(double, Ntot);
    Isr = TMALLOC(double, Ntot); Isi = TMALLOC(double, Ntot);
    Fr = TMALLOC(double, Ntot); Fi = TMALLOC(double, Ntot);
    Kr = TMALLOC(double, Ntot); Ki = TMALLOC(double, Ntot);
    Jr = TMALLOC(double, (size_t)Ntot * (size_t)Ntot);
    Ji = TMALLOC(double, (size_t)Ntot * (size_t)Ntot);
    vsamp = TMALLOC(double, (size_t)N * (size_t)P);

    /* --- source spectrum Is via an OVERSAMPLED least-squares almost-periodic
     * Fourier transform: sample the v=0 source RHS at Nt >> Nh real times
     * t_j = j*dt and solve the normal equations (Gamma^H Gamma) Is = Gamma^H b,
     * with Gamma_{j,h} = exp(j 2pi (k1 f1 + k2 f2) t_j). Oversampling makes
     * Gamma^H Gamma well conditioned (~ Nt*I by near-orthogonality of the
     * harmonics over the equidistributed phases) where a *square* Vandermonde
     * would be catastrophically ill-conditioned beyond a handful of harmonics. */
    {
        int j, k1, k2, a, b, Nt = 6*Nh < 96 ? 96 : 6*Nh;
        double lammax = K1*f1 + K2*f2, dt = 1.0 / (2.1 * lammax);
        double *lam = TMALLOC(double, Nh);
        double *Gr = TMALLOC(double, (size_t)Nt*(size_t)Nh);
        double *Gi = TMALLOC(double, (size_t)Nt*(size_t)Nh);
        double *Mr = TMALLOC(double, (size_t)Nh*(size_t)Nh);
        double *Mi = TMALLOC(double, (size_t)Nh*(size_t)Nh);
        double *cr = TMALLOC(double, (size_t)Nh*(size_t)Nh);
        double *ci = TMALLOC(double, (size_t)Nh*(size_t)Nh);
        double *br = TMALLOC(double, Nh), *bi = TMALLOC(double, Nh);
        double *bsr = TMALLOC(double, (size_t)Nt*(size_t)N);
        a = 0;
        for (k1 = -K1; k1 <= K1; k1++) for (k2 = -K2; k2 <= K2; k2++) { lam[a] = k1*f1 + k2*f2; a++; }
        for (j = 0; j < Nt; j++) {
            for (i = 0; i <= N; i++) { ckt->CKTrhsOld[i] = 0.0; ckt->CKTrhs[i] = 0.0; }
            ckt->CKTtime = (double)j * dt;
            ckt->CKTmode = (ckt->CKTmode & MODEUIC) | MODETRAN | MODEINITTRAN;
            CKTload(ckt);
            for (i = 0; i < N; i++) bsr[(size_t)j*(size_t)N + (size_t)i] = ckt->CKTrhs[i+1];
        }
        for (j = 0; j < Nt; j++)
            for (a = 0; a < Nh; a++) {
                double ang = 2.0*M_PI*lam[a]*(double)j*dt;
                Gr[(size_t)j*(size_t)Nh + (size_t)a] = cos(ang);
                Gi[(size_t)j*(size_t)Nh + (size_t)a] = sin(ang);
            }
        /* M = Gamma^H Gamma :  M_{a,b} = sum_j conj(G_{j,a}) G_{j,b} */
        for (a = 0; a < Nh; a++)
            for (b = 0; b < Nh; b++) {
                double sr = 0, si = 0;
                for (j = 0; j < Nt; j++) {
                    double gar = Gr[(size_t)j*(size_t)Nh+(size_t)a], gai = Gi[(size_t)j*(size_t)Nh+(size_t)a];
                    double gbr = Gr[(size_t)j*(size_t)Nh+(size_t)b], gbi = Gi[(size_t)j*(size_t)Nh+(size_t)b];
                    sr += gar*gbr + gai*gbi;      /* conj(ga)*gb */
                    si += gar*gbi - gai*gbr;
                }
                Mr[(size_t)a*(size_t)Nh+(size_t)b] = sr;
                Mi[(size_t)a*(size_t)Nh+(size_t)b] = si;
            }
        for (i = 0; i < N; i++) {
            for (a = 0; a < Nh; a++) {              /* rhs = Gamma^H b (b real) */
                double sr = 0, si = 0;
                for (j = 0; j < Nt; j++) {
                    double bj = bsr[(size_t)j*(size_t)N + (size_t)i];
                    sr +=  Gr[(size_t)j*(size_t)Nh+(size_t)a] * bj;
                    si += -Gi[(size_t)j*(size_t)Nh+(size_t)a] * bj;
                }
                br[a] = sr; bi[a] = si;
            }
            memcpy(cr, Mr, (size_t)Nh*(size_t)Nh*sizeof(double));
            memcpy(ci, Mi, (size_t)Nh*(size_t)Nh*sizeof(double));
            if (pss_csolve(Nh, cr, ci, br, bi))
                for (a = 0; a < Nh; a++) { br[a] = 0.0; bi[a] = 0.0; }
            for (a = 0; a < Nh; a++) {
                Isr[(size_t)a*(size_t)N + (size_t)i] = br[a];
                Isi[(size_t)a*(size_t)N + (size_t)i] = bi[a];
            }
        }
        FREE(lam); FREE(Gr); FREE(Gi); FREE(Mr); FREE(Mi); FREE(cr); FREE(ci); FREE(br); FREE(bi); FREE(bsr);
    }

    for (i = 0; i < Ntot; i++) { Vr[i] = 0.0; Vi[i] = 0.0; }

    /* --- Newton with E-135 source-stepping continuation --- */
    {
    double lambda = 0.0, dlambda = 1.0, fnorm = 0.0;
    double *Vsr = TMALLOC(double, Ntot), *Vsi = TMALLOC(double, Ntot);
    int iter, nlevels = 0, nnewton = 0, hard_err = 0;
    memcpy(Vsr, Vr, (size_t)Ntot*sizeof(double));
    memcpy(Vsi, Vi, (size_t)Ntot*sizeof(double));

    for (;;) {
        double target = lambda + dlambda;
        int conv = 0;
        if (target > 1.0) target = 1.0;

        for (iter = 0; iter < maxiter; iter++) {
            qp_synth(Vr, Vi, N, K1, K2, P1, P2, vsamp);
            if (qp_extract(ckt, vsamp, N, P1, P2, K1, K2, &hd, IRr, IRi)) {
                fprintf(stderr, "QPSS-HB: device extraction failed.\n");
                rc = E_PARMVAL; hard_err = 1; break;
            }
            hd.f1 = f1; hd.f2 = f2;
            qp_build_matrix(&hd, 0.0, Jr, Ji, 0);
            /* reactive current I_C = (J - Jg)*V (jwC part of J on V) */
            {
                struct qp_harm hg = hd;
                double *Jgr = TMALLOC(double, (size_t)Ntot*(size_t)Ntot);
                double *Jgi = TMALLOC(double, (size_t)Ntot*(size_t)Ntot);
                hg.Cmr = TMALLOC(double, (size_t)hd.nnz*(size_t)hd.Dsz);
                hg.Cmi = TMALLOC(double, (size_t)hd.nnz*(size_t)hd.Dsz);
                qp_build_matrix(&hg, 0.0, Jgr, Jgi, 0);
                FREE(hg.Cmr); FREE(hg.Cmi);
                for (i = 0; i < Ntot; i++) {
                    double xr = 0, xi = 0; int kk;
                    for (kk = 0; kk < Ntot; kk++) {
                        double ar = Jr[(size_t)i*(size_t)Ntot+(size_t)kk] - Jgr[(size_t)i*(size_t)Ntot+(size_t)kk];
                        double ai = Ji[(size_t)i*(size_t)Ntot+(size_t)kk] - Jgi[(size_t)i*(size_t)Ntot+(size_t)kk];
                        xr += ar*Vr[kk] - ai*Vi[kk];
                        xi += ar*Vi[kk] + ai*Vr[kk];
                    }
                    Kr[i] = xr; Ki[i] = xi;
                }
                FREE(Jgr); FREE(Jgi);
            }
            qp_free(&hd);

            for (i = 0; i < Ntot; i++) {
                Fr[i] = IRr[i] + Kr[i] - target*Isr[i];
                Fi[i] = IRi[i] + Ki[i] - target*Isi[i];
            }
            /* Enhancement-178: the settle-mode rhs folded into I_R already
             * carries the DC (operating-point) source values, so -lambda*Is
             * on its own would subtract the DC sources a SECOND time: the
             * converged spectrum answered a doubled DC drive (every DC bias
             * voltage came out exactly 2x, silently corrupting all
             * bias-dependent noise and conversion downstream -- caught by the
             * E-178 cyclostationary referee). Adding the UNSCALED DC source
             * block back makes the net DC drive exactly -lambda*Is_DC.
             * (Assumes the DCOP source value equals the tone-basis DC
             * average, true for DC/SIN sources -- the HB use cases.) */
            { int i00f = K1*(2*K2+1) + K2;
              for (i = 0; i < N; i++) {
                Fr[(size_t)i00f*(size_t)N + (size_t)i] += Isr[(size_t)i00f*(size_t)N + (size_t)i];
                Fi[(size_t)i00f*(size_t)N + (size_t)i] += Isi[(size_t)i00f*(size_t)N + (size_t)i];
              } }
            fnorm = 0.0;
            for (i = 0; i < Ntot; i++)
                fnorm += Fr[i]*Fr[i] + Fi[i]*Fi[i];
            fnorm = sqrt(fnorm);
            nnewton++;
            if (verbose)
                fprintf(stderr, "QPSS-HB lambda=%.4f iter %2d: |F| = %.6e\n", target, iter, fnorm);
            if (isnan(fnorm) || fnorm > 1e300) break;

            for (i = 0; i < Ntot; i++) { Fr[i] = -Fr[i]; Fi[i] = -Fi[i]; }
            if (pss_csolve(Ntot, Jr, Ji, Fr, Fi)) break;
            for (i = 0; i < Ntot; i++) { Vr[i] += Fr[i]; Vi[i] += Fi[i]; }
            if (fnorm < tol) { conv = 1; break; }
        }

        if (hard_err) break;
        if (conv) {
            lambda = target; nlevels++;
            memcpy(Vsr, Vr, (size_t)Ntot*sizeof(double));
            memcpy(Vsi, Vi, (size_t)Ntot*sizeof(double));
            if (lambda >= 1.0 - 1e-9) break;
            dlambda *= 1.7;
            if (lambda + dlambda > 1.0) dlambda = 1.0 - lambda;
        } else {
            memcpy(Vr, Vsr, (size_t)Ntot*sizeof(double));
            memcpy(Vi, Vsi, (size_t)Ntot*sizeof(double));
            dlambda *= 0.5;
            if (dlambda < 1e-5) {
                fprintf(stderr, "QPSS-HB: source stepping stalled at lambda=%.4g "
                                "(|F|=%.3e).\n", lambda, fnorm);
                rc = E_ITERLIM; break;
            }
        }
    }
    if (rc == OK)
        fprintf(stdout, "QPSS-HB: converged in %d iterations, %d continuation step%s "
                        "(|F| = %.3e).\n", nnewton, nlevels, nlevels == 1 ? "" : "s", fnorm);
    FREE(Vsr); FREE(Vsi);
    }

    /* --- output the two-tone spectrum + retain the operating point for qpac --- */
    if (rc == OK) {
        int numNames, error, k1, k2, ord;
        IFuid *nameList = NULL;
        error = CKTnames(ckt, &numNames, &nameList);
        fprintf(stdout,
                "\nQPSS-HB: two-tone steady state (f1 = %g Hz, f2 = %g Hz, "
                "K1 = %d, K2 = %d)\n"
                "  node      (k1,k2)      frequency [Hz]        |V|            phase [deg]\n",
                f1, f2, K1, K2);
        for (i = 0; i < N; i++) {
            const char *nm = (!error && i < numNames) ? (const char *) nameList[i] : "?";
            for (ord = 0; ord <= K1 + K2; ord++)
                for (k1 = -K1; k1 <= K1; k1++)
                    for (k2 = -K2; k2 <= K2; k2++) {
                        double f, sc, vr, vi;
                        if (abs(k1) + abs(k2) != ord) continue;
                        f = k1*f1 + k2*f2;
                        if (f < 0.0) continue;             /* report f >= 0 (conj pairs) */
                        hi = (k1 + K1) * (2*K2 + 1) + (k2 + K2);
                        sc = (f < 1e-9*(f1+f2)) ? 1.0 : 2.0;
                        vr = sc * Vr[(size_t)hi*(size_t)N + (size_t)i];
                        vi = sc * Vi[(size_t)hi*(size_t)N + (size_t)i];
                        fprintf(stdout, "  %-8s  (%2d,%2d)   %16.6e   %14.6e   %10.3f\n",
                                nm, k1, k2, f, hypot(vr, vi),
                                (f < 1e-9*(f1+f2)) ? 0.0 : atan2(vi, vr) * 180.0/M_PI);
                    }
        }
        if (nameList) tfree(nameList);

        /* retain: re-extract at the converged V for a clean conversion structure */
        qp_synth(Vr, Vi, N, K1, K2, P1, P2, vsamp);
        {
            struct qp_harm *sv = TMALLOC(struct qp_harm, 1);
            if (qp_extract(ckt, vsamp, N, P1, P2, K1, K2, sv, IRr, IRi) == 0) {
                sv->f1 = f1; sv->f2 = f2;
                sv->Vr = TMALLOC(double, Ntot); sv->Vi = TMALLOC(double, Ntot);
                memcpy(sv->Vr, Vr, (size_t)Ntot*sizeof(double));
                memcpy(sv->Vi, Vi, (size_t)Ntot*sizeof(double));
                /* capture the AC-source stimulus B0 (a netlist `AC`-flagged source's
                 * RHS -- bias-independent) at the op-point, for qpac (E-137). */
                sv->B0r = TMALLOC(double, N); sv->B0i = TMALLOC(double, N); sv->has_src = 0;
                for (i = 0; i <= N; i++) { ckt->CKTrhsOld[i] = 0.0; ckt->CKTrhs[i] = 0.0; }
                ckt->CKTmode = (ckt->CKTmode & MODEUIC) | MODEDCOP | MODEINITSMSIG;
                CKTload(ckt);
                ckt->CKTomega = 1.0;
                ckt->CKTmode = (ckt->CKTmode & MODEUIC) | MODEAC;
                CKTacLoad(ckt);
                for (i = 0; i < N; i++) {
                    sv->B0r[i] = ckt->CKTrhs[i+1];
                    sv->B0i[i] = ckt->CKTirhs[i+1];
                    if (sv->B0r[i] != 0.0 || sv->B0i[i] != 0.0) sv->has_src = 1;
                }
                if (qpss_hb_saved) { qp_free(qpss_hb_saved); FREE(qpss_hb_saved); }
                qpss_hb_saved = sv;
            } else {
                FREE(sv);
            }
        }
    }

    FREE(Vr); FREE(Vi); FREE(IRr); FREE(IRi); FREE(Isr); FREE(Isi);
    FREE(Fr); FREE(Fi); FREE(Jr); FREE(Ji); FREE(Kr); FREE(Ki); FREE(vsamp);
    return rc;
}


/* ======================================================================
 * Enhancement-137: two-tone small-signal QPAC -- the quasi-periodic
 * analogue of PAC. Around the QPSS operating point retained by
 * `qpss ... hb` (qpss_hb_saved), inject a small signal at f_in; the
 * quasi-periodic operating point converts it to the sidebands
 * f_in + k1*f1 + k2*f2 through the SAME 2-D conversion matrix
 * (qp_build_matrix at f_in), solved by pss_csolve. Same construction
 * as pac_solve_at (E-121/122), on the two-tone harmonic set.
 * ====================================================================== */
int
QPACanalyze(CKTcircuit *ckt, double f_in, int verbose)
{
    struct qp_harm *hd = qpss_hb_saved;
    int    N, Nh, Ntot, i, hi, i00, numNames, error, k1, k2, ord;
    IFuid *nameList = NULL;
    double *Ar, *Ai, *Xr, *Xi;

    if (!hd) {
        fprintf(stderr, "qpac: no QPSS operating point -- run `qpss <expr> <f1> <f2> hb` first.\n");
        return E_NOTFOUND;
    }
    N = hd->N; Nh = hd->Nh; Ntot = hd->Ntot;
    Ar = TMALLOC(double, (size_t)Ntot*(size_t)Ntot);
    Ai = TMALLOC(double, (size_t)Ntot*(size_t)Ntot);
    Xr = TMALLOC(double, Ntot); Xi = TMALLOC(double, Ntot);
    (void) verbose;

    qp_build_matrix(hd, f_in, Ar, Ai, 1);

    /* stimulus in the (0,0) sideband: netlist AC source RHS, or unit current at node 1 */
    memset(Xr, 0, (size_t)Ntot*sizeof(double));
    memset(Xi, 0, (size_t)Ntot*sizeof(double));
    i00 = hd->K1 * (2*hd->K2 + 1) + hd->K2;          /* harmonic index of (0,0) */
    if (hd->has_src) {
        for (i = 0; i < N; i++) {
            Xr[(size_t)i00*(size_t)N + (size_t)i] = hd->B0r[i];
            Xi[(size_t)i00*(size_t)N + (size_t)i] = hd->B0i[i];
        }
    } else {
        Xr[(size_t)i00*(size_t)N + 0] = 1.0;
    }
    if (pss_csolve(Ntot, Ar, Ai, Xr, Xi)) {
        fprintf(stderr, "qpac: singular conversion matrix.\n");
        FREE(Ar); FREE(Ai); FREE(Xr); FREE(Xi);
        return E_SINGULAR;
    }
    FREE(Ar); FREE(Ai);

    error = CKTnames(ckt, &numNames, &nameList);
    fprintf(stdout,
            "\nQPAC: two-tone small-signal response (f_in = %g Hz, f1 = %g Hz, f2 = %g Hz)\n"
            "  node      (k1,k2)   sideband f_in+k1f1+k2f2 [Hz]   |response|      phase [deg]\n",
            f_in, hd->f1, hd->f2);
    for (i = 0; i < N; i++) {
        const char *nm = (!error && i < numNames) ? (const char *) nameList[i] : "?";
        for (ord = 0; ord <= hd->K1 + hd->K2; ord++)
            for (k1 = -hd->K1; k1 <= hd->K1; k1++)
                for (k2 = -hd->K2; k2 <= hd->K2; k2++) {
                    double fsb, xr, xi;
                    if (abs(k1) + abs(k2) != ord) continue;
                    hi = (k1 + hd->K1) * (2*hd->K2 + 1) + (k2 + hd->K2);
                    fsb = f_in + k1*hd->f1 + k2*hd->f2;
                    xr = Xr[(size_t)hi*(size_t)N + (size_t)i];
                    xi = Xi[(size_t)hi*(size_t)N + (size_t)i];
                    fprintf(stdout, "  %-8s  (%2d,%2d)   %18.6e   %14.6e   %10.3f\n",
                            nm, k1, k2, fsb, hypot(xr, xi), atan2(xi, xr) * 180.0/M_PI);
                }
    }
    if (nameList) tfree(nameList);
    FREE(Xr); FREE(Xi);
    return OK;
}


/* ======================================================================
 * Enhancement-138: two-tone small-signal QPnoise (quasi-periodic noise).
 * Around the QPSS operating point retained by `qpss ... hb`, fold every
 * device's noise through the ADJOINT of the 2-D conversion matrix over
 * all sidebands to the output at f_in -- the two-tone analogue of pnoise
 * (E-124). Each device's noise routine computes S*|dTransimp|^2 reading
 * the transimpedance from CKTrhs/CKTirhs, so loading the sideband-(k1,k2)
 * adjoint transfer into CKTrhs and summing over all harmonics folds the
 * noise exactly (mixer/PA folded noise). With no pump the conversion
 * matrix is block-diagonal, so only sideband (0,0) contributes and the
 * result reduces to ordinary .noise.
 * ====================================================================== */

/* Solve the ADJOINT 2-D conversion system H^T Psi = e_{out,(0,0)}. Psi then
 * holds, for every (node j, harmonic hi), the transfer from a unit injection at
 * (j,hi) to the output at sideband (0,0). Reuses qp_build_matrix (forward H) and
 * transposes in place. Returns 0 on success, 1 if singular. */
static int
qp_solve_adjoint(struct qp_harm *hd, double f_in, int outNode, double *Psr, double *Psi)
{
    int    Ntot = hd->Ntot, N = hd->N, i, j, i00, rc;
    double *Ar, *Ai;

    Ar = TMALLOC(double, (size_t)Ntot * (size_t)Ntot);
    Ai = TMALLOC(double, (size_t)Ntot * (size_t)Ntot);
    qp_build_matrix(hd, f_in, Ar, Ai, 1);
    for (i = 0; i < Ntot; i++)                       /* transpose in place: H -> H^T */
        for (j = i + 1; j < Ntot; j++) {
            size_t ij = (size_t)i*(size_t)Ntot + (size_t)j;
            size_t ji = (size_t)j*(size_t)Ntot + (size_t)i;
            double t;
            t = Ar[ij]; Ar[ij] = Ar[ji]; Ar[ji] = t;
            t = Ai[ij]; Ai[ij] = Ai[ji]; Ai[ji] = t;
        }
    memset(Psr, 0, (size_t)Ntot * sizeof(double));
    memset(Psi, 0, (size_t)Ntot * sizeof(double));
    i00 = hd->K1 * (2*hd->K2 + 1) + hd->K2;
    Psr[(size_t)i00 * (size_t)N + (size_t)(outNode - 1)] = 1.0;
    rc = pss_csolve(Ntot, Ar, Ai, Psr, Psi);
    FREE(Ar); FREE(Ai);
    return rc;
}

int
QPnoiseAnalyze(CKTcircuit *ckt, int outNode, double f_in, int cyclo, int verbose)
{
    struct qp_harm *hd = qpss_hb_saved;
    int    N, Nh, Ntot, i, j, hi, i00;
    double *Psr, *Psi, *Xr, *Xi;
    double onoise = 0.0, inoise, gain2 = 1.0;
    NOISEAN nj;
    Ndata   data;
    JOB    *oldJob;

    if (!hd) {
        fprintf(stderr, "qpnoise: no QPSS operating point -- run `qpss <expr> <f1> <f2> hb` first.\n");
        return E_NOTFOUND;
    }
    N = hd->N; Nh = hd->Nh; Ntot = hd->Ntot;
    if (outNode <= 0 || outNode > N) {
        fprintf(stderr, "qpnoise: bad output node.\n");
        return E_PARMVAL;
    }
    i00 = hd->K1 * (2*hd->K2 + 1) + hd->K2;

    /* bias the devices at the QPSS operating point (phase (0,0) sample: v = sum of
     * all harmonic coefficients) so each noise PSD is at the periodic bias. */
    for (j = 1; j <= N; j++) {
        double v = 0.0;
        for (hi = 0; hi < Nh; hi++) v += hd->Vr[(size_t)hi*(size_t)N + (size_t)(j-1)];
        ckt->CKTrhsOld[j] = v;
    }
    ckt->CKTrhsOld[0] = 0.0;
    ckt->CKTmode = (ckt->CKTmode & MODEUIC) | MODEDCOP | MODEINITSMSIG;
    CKTload(ckt);

    /* minimal NOISEAN context (device noise routines cast CKTcurJob to NOISEAN*) */
    memset(&nj, 0, sizeof(nj));
    nj.NstartFreq = f_in;
    nj.NstopFreq = f_in;
    nj.NnumSteps = 1;
    nj.NstpType = 0;
    nj.NStpsSm = 0;
    nj.JOBname = "qpnoise";
    memset(&data, 0, sizeof(data));
    data.prtSummary = FALSE;
    oldJob = ckt->CKTcurJob;
    ckt->CKTcurJob = (JOB *) &nj;
    for (i = 0; i < DEVmaxnum; i++)
        if (DEVices[i] && DEVices[i]->DEVnoise && ckt->CKThead[i]) {
            double dummy = 0.0;
            DEVices[i]->DEVnoise(N_DENS, N_OPEN, ckt->CKThead[i], ckt, &data, &dummy);
        }

    Psr = TMALLOC(double, Ntot); Psi = TMALLOC(double, Ntot);
    Xr  = TMALLOC(double, Ntot); Xi  = TMALLOC(double, Ntot);

    /* adjoint transfer from every (node, harmonic) to the output at (0,0) */
    data.freq = f_in; data.delFreq = 0.0; data.prtSummary = FALSE;
    if (qp_solve_adjoint(hd, f_in, outNode, Psr, Psi) == 0) {
        if (!cyclo) {
            /* STATIONARY (E-138): fold S*|Psi_{(k1,k2)}|^2 over all sidebands, with
             * the device PSD S taken once at the operating-point bias. */
            for (hi = 0; hi < Nh; hi++) {
                double dens = 0.0;
                size_t blk = (size_t)hi * (size_t)N;
                /* Enhancement-177: evaluate the source PSD at the harmonic's
                 * own physical frequency (see pnoise_sweep). */
                data.freq = fabs(f_in + hd->h1[hi] * hd->f1 + hd->h2[hi] * hd->f2);
                if (data.freq == 0.0)
                    data.freq = f_in;
                for (j = 1; j <= N; j++) {
                    ckt->CKTrhs[j]  = Psr[blk + (size_t)(j-1)];
                    ckt->CKTirhs[j] = Psi[blk + (size_t)(j-1)];
                }
                ckt->CKTrhs[0] = 0.0; ckt->CKTirhs[0] = 0.0;
                for (i = 0; i < DEVmaxnum; i++)
                    if (DEVices[i] && DEVices[i]->DEVnoise && ckt->CKThead[i])
                        DEVices[i]->DEVnoise(N_DENS, N_CALC, ckt->CKThead[i], ckt, &data, &dens);
                onoise += dens;                   /* sum device noise over sidebands */
            }
        } else {
            /* CYCLOSTATIONARY (E-139, exact-separable E-178): the device PSD
             * swings over the two-tone quasi-period. E-139 used the identity
             * onoise = (1/P) Sum_s S(t_s)|A_s|^2, exact only for frequency-FLAT
             * (modulated-white) sources. E-178 replaces it with the exact
             * separable model, the 2-D analog of the single-tone pnoise cyclo
             * path (see pnoise_sweep): per generator g,
             *   onoise = Sum_{q1,q2} |B_{q1,q2}(g)|^2 * g_g(|f + q1 f1 + q2 f2|),
             *   B_q(g) = (1/P1P2) Sum_s m_g(t_s) dA_s(g) e^{+j(q1 th1 + q2 th2)},
             * with per-generator amplitudes recovered by load POLARIZATION
             * against the reference load R = Psi_(0,0) (five DEVnoise sweeps
             * per sample) and the spectral shape g_g MEASURED pointwise at the
             * folded frequencies. Flat generators (thermal/shot) are routed
             * through the original identity (exact for any quadratic form);
             * colored ones (flicker, noise_table) through the B_q machinery.
             * Stationary m: reduces to the E-177 stationary sum; flat g:
             * Parseval collapses back to the E-139 identity. */
            int P1 = hd->P1, P2 = hd->P2, s1, s2;
            int Ptot = P1 * P2;
            int Q1 = 2 * hd->K1, Q2 = 2 * hd->K2;
            int nq = (2*Q1 + 1) * (2*Q2 + 1), qi, q1, q2;
            int G, g, sw, b1, b2;
            int NB1 = (P1 < 4) ? P1 : 4, NB2 = (P2 < 4) ? P2 : 4;
            double *vsamp = TMALLOC(double, (size_t)N * (size_t)Ptot);
            double *bset  = TMALLOC(double, N);
            double *Lr    = TMALLOC(double, N + 1);
            double *Li    = TMALLOC(double, N + 1);
            NOISEAN nj2;
            Ndata   d2;
            int    *is_total;
            double *swv[5], *Bqr, *Bqi, *Dflat, *zsum, *probe;
            double  dummy_dens;

            qp_synth(hd->Vr, hd->Vi, N, hd->K1, hd->K2, P1, P2, vsamp);

            /* generator naming pass (one summary slot per noise generator) */
            nj2 = nj;
            nj2.NStpsSm = 1;
            memset(&d2, 0, sizeof d2);
            ckt->CKTcurJob = (JOB *) &nj2;
            for (i = 0; i < DEVmaxnum; i++)
                if (DEVices[i] && DEVices[i]->DEVnoise && ckt->CKThead[i]) {
                    dummy_dens = 0.0;
                    DEVices[i]->DEVnoise(N_DENS, N_OPEN, ckt->CKThead[i], ckt, &d2, &dummy_dens);
                }
            G = d2.numPlots;
            is_total = TMALLOC(int, (G > 0) ? G : 1);
            for (g = 0; g < G; g++) {
                /* family-TOTAL slots: name is a proper prefix of the preceding
                 * generator's name (see pnoise_sweep) */
                is_total[g] = 0;
                if (g > 0) {
                    const char *nm = (const char *) d2.namelist[g];
                    const char *pm = (const char *) d2.namelist[g - 1];
                    size_t ln = strlen(nm);
                    if (strncmp(nm, pm, ln) == 0 && strlen(pm) > ln)
                        is_total[g] = 1;
                }
            }
            d2.outpVector = TMALLOC(double, (G > 0) ? G : 1);
            d2.prtSummary = TRUE;
            d2.delFreq = 0.0;
            for (i = 0; i < 5; i++)
                swv[i] = TMALLOC(double, (G > 0) ? G : 1);
            Bqr   = TMALLOC(double, (size_t)G * (size_t)nq);
            Bqi   = TMALLOC(double, (size_t)G * (size_t)nq);
            Dflat = TMALLOC(double, (size_t)G);
            zsum  = TMALLOC(double, (size_t)G);
            probe = TMALLOC(double, (size_t)NB1 * (size_t)NB2 * (size_t)G * (size_t)nq);
            memset(Bqr, 0, (size_t)G * (size_t)nq * sizeof(double));
            memset(Bqi, 0, (size_t)G * (size_t)nq * sizeof(double));
            memset(Dflat, 0, (size_t)G * sizeof(double));
            memset(zsum, 0, (size_t)G * sizeof(double));
            memset(probe, 0, (size_t)NB1 * (size_t)NB2 * (size_t)G * (size_t)nq * sizeof(double));

            for (s1 = 0; s1 < P1; s1++)
                for (s2 = 0; s2 < P2; s2++) {
                    int s = s1 * P2 + s2;
                    double th1 = 2.0*M_PI*(double)s1/(double)P1;
                    double th2 = 2.0*M_PI*(double)s2/(double)P2;
                    /* bias the devices at this sample's quasi-periodic operating point */
                    for (j = 1; j <= N; j++)
                        ckt->CKTrhsOld[j] = vsamp[(size_t)s*(size_t)N + (size_t)(j-1)];
                    ckt->CKTrhsOld[0] = 0.0;
                    /* settle limited junctions at the fixed sample voltages (E-134) so a
                     * diode/BJT/MOS reports the noise PSD at THIS sample's bias, not a
                     * stale stored junction -- else the noise looks stationary. */
                    ckt->CKTmode = (ckt->CKTmode & MODEUIC) | MODEDCOP | MODEINITFLOAT;
                    { int inner;
                      for (inner = 0; inner < 100; inner++) {
                        double bnorm = 0.0, dnorm = 0.0;
                        for (i = 0; i <= N; i++) ckt->CKTrhs[i] = 0.0;
                        CKTload(ckt);
                        for (i = 1; i <= N; i++) {
                            double db = ckt->CKTrhs[i] - bset[i-1];
                            dnorm += db*db; bnorm += ckt->CKTrhs[i]*ckt->CKTrhs[i];
                            bset[i-1] = ckt->CKTrhs[i];
                        }
                        if (inner > 0 && sqrt(dnorm) <= 1e-12*(sqrt(bnorm)+1e-30)) break;
                      } }
                    ckt->CKTmode = (ckt->CKTmode & MODEUIC) | MODEDCOP | MODEINITSMSIG;
                    CKTload(ckt);
                    /* A-_s(j) = Sum_{(k1,k2)} Psi_{(k1,k2)}(j) e^{-j(k1 th1 + k2 th2)} */
                    for (j = 1; j <= N; j++) {
                        double ar = 0.0, ai = 0.0;
                        for (hi = 0; hi < Nh; hi++) {
                            size_t idx = (size_t)hi*(size_t)N + (size_t)(j-1);
                            double ang = (double)hd->h1[hi]*th1 + (double)hd->h2[hi]*th2;
                            double cs = cos(ang), sn = sin(ang);
                            ar += Psr[idx]*cs + Psi[idx]*sn;
                            ai += Psi[idx]*cs - Psr[idx]*sn;
                        }
                        Lr[j] = ar; Li[j] = ai;
                    }
                    for (sw = 0; sw < 5; sw++) {
                        size_t r0 = (size_t)i00 * (size_t)N;   /* Psi_(0,0) = R */
                        for (j = 1; j <= N; j++) {
                            double rr = Psr[r0 + (size_t)(j-1)];
                            double ri = Psi[r0 + (size_t)(j-1)];
                            switch (sw) {
                            case 0: ckt->CKTrhs[j] = Lr[j] + rr;  ckt->CKTirhs[j] = Li[j] + ri;  break;
                            case 1: ckt->CKTrhs[j] = Lr[j] - rr;  ckt->CKTirhs[j] = Li[j] - ri;  break;
                            case 2: ckt->CKTrhs[j] = Lr[j] - ri;  ckt->CKTirhs[j] = Li[j] + rr;  break; /* A + jR */
                            case 3: ckt->CKTrhs[j] = Lr[j] + ri;  ckt->CKTirhs[j] = Li[j] - rr;  break; /* A - jR */
                            default: ckt->CKTrhs[j] = rr;         ckt->CKTirhs[j] = ri;          break; /* R */
                            }
                        }
                        ckt->CKTrhs[0] = 0.0;  ckt->CKTirhs[0] = 0.0;
                        d2.freq = f_in;  d2.outNumber = 0;
                        dummy_dens = 0.0;
                        for (i = 0; i < DEVmaxnum; i++)
                            if (DEVices[i] && DEVices[i]->DEVnoise && ckt->CKThead[i])
                                DEVices[i]->DEVnoise(N_DENS, N_CALC, ckt->CKThead[i],
                                                     ckt, &d2, &dummy_dens);
                        memcpy(swv[sw], d2.outpVector, (size_t)G * sizeof(double));
                    }
                    for (g = 0; g < G; g++) {
                        double dp, dm, djp, djm, z, qa;
                        if (is_total[g])
                            continue;
                        dp = swv[0][g];  dm = swv[1][g];
                        djp = swv[2][g]; djm = swv[3][g];
                        z = swv[4][g];
                        qa = 0.5 * (dp + dm) - z;
                        Dflat[g] += qa;
                        if (z > 1e-280) {
                            double yr = 0.25 * (dp - dm);
                            double yi = 0.25 * (djp - djm);
                            double sz = sqrt(z);
                            double cr = yr / sz, ci = yi / sz;
                            zsum[g] += z;
                            for (qi = 0; qi < nq; qi++) {
                                double aq;
                                q1 = qi / (2*Q2 + 1) - Q1;
                                q2 = qi % (2*Q2 + 1) - Q2;
                                aq = (double)q1*th1 + (double)q2*th2;
                                Bqr[(size_t)g*(size_t)nq + (size_t)qi] += cr*cos(aq) - ci*sin(aq);
                                Bqi[(size_t)g*(size_t)nq + (size_t)qi] += cr*sin(aq) + ci*cos(aq);
                            }
                        }
                    }
                }

            /* spectral-shape probes at the folded frequencies, load R */
            for (b1 = 0; b1 < NB1; b1++)
                for (b2 = 0; b2 < NB2; b2++) {
                    int sb = (b1 * P1 / NB1) * P2 + (b2 * P2 / NB2);
                    size_t r0 = (size_t)i00 * (size_t)N;
                    for (j = 1; j <= N; j++)
                        ckt->CKTrhsOld[j] = vsamp[(size_t)sb*(size_t)N + (size_t)(j-1)];
                    ckt->CKTrhsOld[0] = 0.0;
                    ckt->CKTmode = (ckt->CKTmode & MODEUIC) | MODEDCOP | MODEINITFLOAT;
                    { int inner;
                      for (inner = 0; inner < 100; inner++) {
                        double bnorm = 0.0, dnorm = 0.0;
                        for (i = 0; i <= N; i++) ckt->CKTrhs[i] = 0.0;
                        CKTload(ckt);
                        for (i = 1; i <= N; i++) {
                            double db = ckt->CKTrhs[i] - bset[i-1];
                            dnorm += db*db; bnorm += ckt->CKTrhs[i]*ckt->CKTrhs[i];
                            bset[i-1] = ckt->CKTrhs[i];
                        }
                        if (inner > 0 && sqrt(dnorm) <= 1e-12*(sqrt(bnorm)+1e-30)) break;
                      } }
                    ckt->CKTmode = (ckt->CKTmode & MODEUIC) | MODEDCOP | MODEINITSMSIG;
                    CKTload(ckt);
                    for (j = 1; j <= N; j++) {
                        ckt->CKTrhs[j]  = Psr[r0 + (size_t)(j-1)];
                        ckt->CKTirhs[j] = Psi[r0 + (size_t)(j-1)];
                    }
                    ckt->CKTrhs[0] = 0.0;  ckt->CKTirhs[0] = 0.0;
                    for (qi = 0; qi < nq; qi++) {
                        double fq;
                        q1 = qi / (2*Q2 + 1) - Q1;
                        q2 = qi % (2*Q2 + 1) - Q2;
                        fq = fabs(f_in + (double)q1*hd->f1 + (double)q2*hd->f2);
                        if (fq == 0.0)
                            fq = f_in;
                        d2.freq = fq;  d2.outNumber = 0;
                        dummy_dens = 0.0;
                        for (i = 0; i < DEVmaxnum; i++)
                            if (DEVices[i] && DEVices[i]->DEVnoise && ckt->CKThead[i])
                                DEVices[i]->DEVnoise(N_DENS, N_CALC, ckt->CKThead[i],
                                                     ckt, &d2, &dummy_dens);
                        for (g = 0; g < G; g++)
                            probe[(((size_t)(b1*NB2 + b2)) * (size_t)G + (size_t)g)
                                  * (size_t)nq + (size_t)qi] = d2.outpVector[g];
                    }
                }

            /* combine */
            for (g = 0; g < G; g++) {
                double zref = 0.0, on_g;
                int    b, bbest = 0, colored = 0;
                int    q00 = Q1 * (2*Q2 + 1) + Q2;             /* (0,0) bin */
                if (is_total[g])
                    continue;
                for (b = 0; b < NB1*NB2; b++) {
                    double zr = probe[((size_t)b * (size_t)G + (size_t)g) * (size_t)nq
                                      + (size_t)q00];
                    if (zr > zref) { zref = zr; bbest = b; }
                }
                if (zref > 0.0) {
                    for (qi = 0; qi < nq; qi++) {
                        double rq = probe[((size_t)bbest * (size_t)G + (size_t)g) * (size_t)nq
                                          + (size_t)qi] / zref;
                        if (fabs(rq - 1.0) > 1e-6) { colored = 1; break; }
                    }
                }
                if (colored && zsum[g] > 0.0) {
                    on_g = 0.0;
                    for (qi = 0; qi < nq; qi++) {
                        double rq = probe[((size_t)bbest * (size_t)G + (size_t)g) * (size_t)nq
                                          + (size_t)qi] / zref;
                        double br = Bqr[(size_t)g*(size_t)nq + (size_t)qi] / (double)Ptot;
                        double bi = Bqi[(size_t)g*(size_t)nq + (size_t)qi] / (double)Ptot;
                        on_g += (br*br + bi*bi) * rq;
                    }
                } else {
                    on_g = Dflat[g] / (double)Ptot;
                }
                onoise += on_g;
            }
            ckt->CKTcurJob = (JOB *) &nj;
            for (i = 0; i < 5; i++) FREE(swv[i]);
            FREE(is_total); FREE(d2.outpVector); FREE(d2.namelist);
            FREE(Bqr); FREE(Bqi); FREE(Dflat); FREE(zsum); FREE(probe);
            FREE(vsamp); FREE(bset); FREE(Lr); FREE(Li);
        }
    }

    /* input-referred: divide by the source->output conversion gain at (0,0) */
    if (hd->has_src) {
        double *Ar = TMALLOC(double, (size_t)Ntot*(size_t)Ntot);
        double *Ai = TMALLOC(double, (size_t)Ntot*(size_t)Ntot);
        qp_build_matrix(hd, f_in, Ar, Ai, 1);
        memset(Xr, 0, (size_t)Ntot*sizeof(double));
        memset(Xi, 0, (size_t)Ntot*sizeof(double));
        for (i = 0; i < N; i++) {
            Xr[(size_t)i00*(size_t)N + (size_t)i] = hd->B0r[i];
            Xi[(size_t)i00*(size_t)N + (size_t)i] = hd->B0i[i];
        }
        if (pss_csolve(Ntot, Ar, Ai, Xr, Xi) == 0) {
            size_t oidx = (size_t)i00*(size_t)N + (size_t)(outNode-1);
            gain2 = Xr[oidx]*Xr[oidx] + Xi[oidx]*Xi[oidx];
        }
        FREE(Ar); FREE(Ai);
    }
    inoise = onoise / MAX(gain2, N_MINGAIN);

    ckt->CKTcurJob = oldJob;
    (void) verbose;
    fprintf(stdout,
            "\nQPnoise: two-tone %s output noise at f_in = %g Hz (folding %d sidebands, "
            "f1 = %g, f2 = %g)\n"
            "  onoise density = %.6e  V^2/Hz   (%.6e V/sqrt(Hz))\n"
            "  inoise density = %.6e           (gain^2 = %.6e)\n",
            cyclo ? "cyclostationary" : "stationary",
            f_in, Nh, hd->f1, hd->f2, onoise, sqrt(onoise), inoise, gain2);

    FREE(Psr); FREE(Psi); FREE(Xr); FREE(Xi);
    return OK;
}


/* ======================================================================
 * Enhancement-141: two-tone small-signal QPXF -- the quasi-periodic transfer
 * function, the ADJOINT of QPAC (Enhancement-137). One adjoint solve of the
 * 2-D conversion matrix, H^T Psi = e_{out,(0,0)}, gives the transimpedance
 * Psi_{(k1,k2),j} from an injection at (node j, sideband (k1,k2)) to the
 * output at sideband (0,0). Dotting each sideband block of Psi with the
 * netlist AC-source pattern B0 gives the transfer from that input sideband to
 * the output. By reciprocity the (0,0) transfer equals the QPAC response at
 * the output node -- a cross-check that pins the adjoint solve (cf. PXF/PAC,
 * Enhancement-125). Reuses qp_solve_adjoint (built for QPnoise, E-138).
 * ====================================================================== */
int
QPXFanalyze(CKTcircuit *ckt, int outNode, double f_in, int verbose)
{
    struct qp_harm *hd = qpss_hb_saved;
    int    N, Ntot, j, hi, numNames, error, k1, k2, ord;
    IFuid *nameList = NULL;
    double *Psr, *Psi;

    if (!hd) {
        fprintf(stderr, "qpxf: no QPSS operating point -- run `qpss <expr> <f1> <f2> hb` first.\n");
        return E_NOTFOUND;
    }
    N = hd->N; Ntot = hd->Ntot;
    if (outNode <= 0 || outNode > N) { fprintf(stderr, "qpxf: bad output node.\n"); return E_PARMVAL; }
    (void) verbose;
    Psr = TMALLOC(double, Ntot); Psi = TMALLOC(double, Ntot);

    if (qp_solve_adjoint(hd, f_in, outNode, Psr, Psi)) {
        fprintf(stderr, "qpxf: singular conversion matrix.\n");
        FREE(Psr); FREE(Psi); return E_SINGULAR;
    }

    error = CKTnames(ckt, &numNames, &nameList);
    fprintf(stdout,
            "\nQPXF: two-tone transfer function to node %s (f_in = %g Hz, f1 = %g Hz, f2 = %g Hz)\n"
            "  (k1,k2)   input sideband f_in+k1f1+k2f2 [Hz]   |H|            phase [deg]\n",
            (!error && (outNode-1) < numNames) ? (const char *) nameList[outNode-1] : "?",
            f_in, hd->f1, hd->f2);
    for (ord = 0; ord <= hd->K1 + hd->K2; ord++)
        for (k1 = -hd->K1; k1 <= hd->K1; k1++)
            for (k2 = -hd->K2; k2 <= hd->K2; k2++) {
                double fsb, hr = 0.0, hii = 0.0;
                if (abs(k1) + abs(k2) != ord) continue;
                hi = (k1 + hd->K1) * (2*hd->K2 + 1) + (k2 + hd->K2);
                /* transfer from input at this sideband to the output: Psi_block . B0
                 * (AC-source pattern), or Psi at the injection node if no AC source */
                if (hd->has_src) {
                    for (j = 0; j < N; j++) {
                        double pr = Psr[(size_t)hi*(size_t)N + (size_t)j];
                        double pi = Psi[(size_t)hi*(size_t)N + (size_t)j];
                        hr += pr*hd->B0r[j] - pi*hd->B0i[j];
                        hii += pr*hd->B0i[j] + pi*hd->B0r[j];
                    }
                } else {
                    hr = Psr[(size_t)hi*(size_t)N + 0];
                    hii = Psi[(size_t)hi*(size_t)N + 0];
                }
                fsb = f_in + k1*hd->f1 + k2*hd->f2;
                fprintf(stdout, "  (%2d,%2d)   %18.6e   %14.6e   %10.3f\n",
                        k1, k2, fsb, hypot(hr, hii), atan2(hii, hr) * 180.0/M_PI);
            }
    if (nameList) tfree(nameList);
    FREE(Psr); FREE(Psi);
    return OK;
}


/* ======================================================================
 * Enhancement-142: input-frequency SWEEP for the two-tone small-signal
 * analyses. `qpac`/`qpnoise`/`qpxf` gain a dec|oct|lin sweep of f_in. Each
 * point reuses the same single-frequency solve; these fill caller arrays
 * (freqs[] + data[], point-major) and the frontend builds the ngspice plot
 * (conversion gain / NF / image-rejection curves vs f_in). Return #points.
 * ====================================================================== */

/* QPAC swept: data[pt*N + node] = |(0,0) response| per node. */
int
QPACsweep(CKTcircuit *ckt, int stepType, int np, double fstart, double fstop, double *freqs, double *data)
{
    struct qp_harm *hd = qpss_hb_saved;
    int    N, Ntot, i, i00, pt = 0;
    double *Ar, *Ai, *Xr, *Xi, freq, mult, linstep;
    (void) ckt;
    if (!hd) { fprintf(stderr, "qpac: no QPSS operating point -- run `qpss ... hb` first.\n"); return -1; }
    N = hd->N; Ntot = hd->Ntot; i00 = hd->K1*(2*hd->K2+1) + hd->K2;
    Ar = TMALLOC(double,(size_t)Ntot*(size_t)Ntot); Ai = TMALLOC(double,(size_t)Ntot*(size_t)Ntot);
    Xr = TMALLOC(double, Ntot); Xi = TMALLOC(double, Ntot);
    mult = (stepType==1)?pow(10.0,1.0/np):(stepType==2)?pow(2.0,1.0/np):0.0;
    linstep = (np>1)?(fstop-fstart)/(np-1):0.0;
    for (freq=fstart; freq<=fstop*(1.0+1e-9); ) {
        qp_build_matrix(hd, freq, Ar, Ai, 1);
        memset(Xr,0,(size_t)Ntot*sizeof(double)); memset(Xi,0,(size_t)Ntot*sizeof(double));
        if (hd->has_src) for (i=0;i<N;i++){ Xr[(size_t)i00*(size_t)N+(size_t)i]=hd->B0r[i]; Xi[(size_t)i00*(size_t)N+(size_t)i]=hd->B0i[i]; }
        else Xr[(size_t)i00*(size_t)N+0]=1.0;
        if (pss_csolve(Ntot,Ar,Ai,Xr,Xi)) for (i=0;i<Ntot;i++){Xr[i]=0;Xi[i]=0;}
        freqs[pt]=freq;
        for (i=0;i<N;i++){ double xr=Xr[(size_t)i00*(size_t)N+(size_t)i], xi=Xi[(size_t)i00*(size_t)N+(size_t)i]; data[(size_t)pt*(size_t)N+(size_t)i]=hypot(xr,xi); }
        pt++;
        if (stepType==0){ if(np<=1) break; freq+=linstep; } else freq*=mult;
    }
    FREE(Ar); FREE(Ai); FREE(Xr); FREE(Xi);
    return pt;
}

/* QPnoise swept: data[pt*2+0]=onoise, data[pt*2+1]=inoise. */
int
QPnoiseSweep(CKTcircuit *ckt, int outNode, int stepType, int np, double fstart, double fstop, double *freqs, double *data)
{
    struct qp_harm *hd = qpss_hb_saved;
    int    N, Nh, Ntot, i, j, k, i00, pt = 0;
    NOISEAN nj; Ndata dn; JOB *oldJob;
    double *Psr, *Psi, *Ar, *Ai, *Xr, *Xi, freq, mult, linstep;
    if (!hd) { fprintf(stderr, "qpnoise: no QPSS operating point -- run `qpss ... hb` first.\n"); return -1; }
    N = hd->N; Nh = hd->Nh; Ntot = hd->Ntot; i00 = hd->K1*(2*hd->K2+1) + hd->K2;
    if (outNode <= 0 || outNode > N) { fprintf(stderr, "qpnoise: bad output node.\n"); return -1; }
    for (j=1;j<=N;j++){ double v=0.0; for (k=0;k<Nh;k++) v+=hd->Vr[(size_t)k*(size_t)N+(size_t)(j-1)]; ckt->CKTrhsOld[j]=v; }
    ckt->CKTrhsOld[0]=0.0; ckt->CKTmode=(ckt->CKTmode & MODEUIC)|MODEDCOP|MODEINITSMSIG; CKTload(ckt);
    memset(&nj,0,sizeof(nj)); nj.NstartFreq=fstart; nj.NstopFreq=fstop; nj.NnumSteps=np; nj.NstpType=stepType; nj.NStpsSm=0; nj.JOBname="qpnoise";
    memset(&dn,0,sizeof(dn)); dn.prtSummary=FALSE;
    oldJob=ckt->CKTcurJob; ckt->CKTcurJob=(JOB*)&nj;
    for (i=0;i<DEVmaxnum;i++) if (DEVices[i]&&DEVices[i]->DEVnoise&&ckt->CKThead[i]){ double d=0.0; DEVices[i]->DEVnoise(N_DENS,N_OPEN,ckt->CKThead[i],ckt,&dn,&d); }
    Psr=TMALLOC(double,Ntot); Psi=TMALLOC(double,Ntot);
    Ar=TMALLOC(double,(size_t)Ntot*(size_t)Ntot); Ai=TMALLOC(double,(size_t)Ntot*(size_t)Ntot);
    Xr=TMALLOC(double,Ntot); Xi=TMALLOC(double,Ntot);
    mult=(stepType==1)?pow(10.0,1.0/np):(stepType==2)?pow(2.0,1.0/np):0.0;
    linstep=(np>1)?(fstop-fstart)/(np-1):0.0;
    for (freq=fstart; freq<=fstop*(1.0+1e-9); ) {
        double onoise=0.0, gain2=1.0;
        if (qp_solve_adjoint(hd, freq, outNode, Psr, Psi)==0) {
            dn.freq=freq; dn.delFreq=0.0; dn.prtSummary=FALSE;
            for (k=0;k<Nh;k++){ double dens=0.0; size_t blk=(size_t)k*(size_t)N;
                for (j=1;j<=N;j++){ ckt->CKTrhs[j]=Psr[blk+(size_t)(j-1)]; ckt->CKTirhs[j]=Psi[blk+(size_t)(j-1)]; }
                ckt->CKTrhs[0]=0.0; ckt->CKTirhs[0]=0.0;
                for (i=0;i<DEVmaxnum;i++) if (DEVices[i]&&DEVices[i]->DEVnoise&&ckt->CKThead[i]) DEVices[i]->DEVnoise(N_DENS,N_CALC,ckt->CKThead[i],ckt,&dn,&dens);
                onoise+=dens;
            }
        }
        if (hd->has_src){
            qp_build_matrix(hd, freq, Ar, Ai, 1);
            memset(Xr,0,(size_t)Ntot*sizeof(double)); memset(Xi,0,(size_t)Ntot*sizeof(double));
            for (i=0;i<N;i++){ Xr[(size_t)i00*(size_t)N+(size_t)i]=hd->B0r[i]; Xi[(size_t)i00*(size_t)N+(size_t)i]=hd->B0i[i]; }
            if (pss_csolve(Ntot,Ar,Ai,Xr,Xi)==0){ size_t o=(size_t)i00*(size_t)N+(size_t)(outNode-1); gain2=Xr[o]*Xr[o]+Xi[o]*Xi[o]; }
        }
        freqs[pt]=freq; data[(size_t)pt*2+0]=onoise; data[(size_t)pt*2+1]=onoise/MAX(gain2,N_MINGAIN);
        pt++;
        if (stepType==0){ if(np<=1) break; freq+=linstep; } else freq*=mult;
    }
    ckt->CKTcurJob=oldJob;
    FREE(Psr);FREE(Psi);FREE(Ar);FREE(Ai);FREE(Xr);FREE(Xi);
    return pt;
}

/* QPXF swept: data[pt*2+0]=|(0,0) transfer|, data[pt*2+1]=total conversion. */
int
QPXFsweep(CKTcircuit *ckt, int outNode, int stepType, int np, double fstart, double fstop, double *freqs, double *data)
{
    struct qp_harm *hd = qpss_hb_saved;
    int    N, Nh, Ntot, j, hi, i00, pt = 0;
    double *Psr, *Psi, freq, mult, linstep;
    (void) ckt;
    if (!hd) { fprintf(stderr, "qpxf: no QPSS operating point -- run `qpss ... hb` first.\n"); return -1; }
    N = hd->N; Nh = hd->Nh; Ntot = hd->Ntot; i00 = hd->K1*(2*hd->K2+1) + hd->K2;
    if (outNode <= 0 || outNode > N) { fprintf(stderr, "qpxf: bad output node.\n"); return -1; }
    Psr=TMALLOC(double,Ntot); Psi=TMALLOC(double,Ntot);
    mult=(stepType==1)?pow(10.0,1.0/np):(stepType==2)?pow(2.0,1.0/np):0.0;
    linstep=(np>1)?(fstop-fstart)/(np-1):0.0;
    for (freq=fstart; freq<=fstop*(1.0+1e-9); ) {
        double xf0=0.0, conv=0.0;
        if (qp_solve_adjoint(hd, freq, outNode, Psr, Psi)==0){
            for (hi=0;hi<Nh;hi++){
                double hr=0.0, hii=0.0, m2;
                if (hd->has_src) for (j=0;j<N;j++){ double pr=Psr[(size_t)hi*(size_t)N+(size_t)j], pi=Psi[(size_t)hi*(size_t)N+(size_t)j]; hr+=pr*hd->B0r[j]-pi*hd->B0i[j]; hii+=pr*hd->B0i[j]+pi*hd->B0r[j]; }
                else { hr=Psr[(size_t)hi*(size_t)N+0]; hii=Psi[(size_t)hi*(size_t)N+0]; }
                m2=hr*hr+hii*hii;
                if (hi==i00) xf0=sqrt(m2); else conv+=m2;
            }
        }
        freqs[pt]=freq; data[(size_t)pt*2+0]=xf0; data[(size_t)pt*2+1]=sqrt(conv);
        pt++;
        if (stepType==0){ if(np<=1) break; freq+=linstep; } else freq*=mult;
    }
    FREE(Psr); FREE(Psi);
    return pt;
}


/* ======================================================================
 * Enhancement-140: oscillator phase noise.
 *
 * (1) HBOSCanalyze -- AUTONOMOUS harmonic balance. An oscillator has no
 *     driving source, so the HB residual F(V) = I_R + [dq/dt] = 0 is solved
 *     for the harmonics V *and* the unknown oscillation frequency w0, with a
 *     phase gauge (the solution's phase is free). The Jacobian dF/dV is the
 *     conversion matrix H -- SINGULAR, its right null space the phase mode
 *     u_k = jk V_k (shifting the oscillator's phase is a symmetry). Newton on
 *     the bordered system  [ H  dF/dw0 ; u*^T  0 ] [dV; dw0] = [-F; 0]
 *     (nonsingular by bordering) refines (V, w0) from a transient seed.
 * (2) PhaseNoiseAnalyze -- the phase-noise spectrum L(df). The PPV (Demir's
 *     perturbation projection vector) is the LEFT null vector of H, normalized
 *     so <v1, phase-mode> = 1. The phase diffusion constant c is the device
 *     noise folded through the PPV (same machinery as pnoise, transfer -> PPV),
 *     and L(df) = 10 log10( f0^2 c / df^2 ) -- the classic 1/df^2 (-20 dB/dec)
 *     oscillator phase noise, saturating into a Lorentzian near the carrier.
 * ====================================================================== */

static struct pac_harm osc_hd;                 /* retained oscillator conversion data */
static double *osc_Vr = NULL, *osc_Vi = NULL;  /* [Ntot] retained oscillator spectrum */
static double  osc_f0 = 0.0;
static int     osc_node = 0, osc_valid = 0;

int
HBOSCanalyze(CKTcircuit *ckt, int oscNode, int K, int Pin, double f0seed,
             double ampseed, int maxiter, double tol, int verbose)
{
    int N = SMPmatSize(ckt->CKTmatrix);
    int P = Pin > 0 ? Pin : ((8*K < 32) ? 32 : 8*K);
    int Ntot = (2*K+1)*N, Naug = Ntot + 1;
    int iter, s, i, k, c, rc = E_ITERLIM, have_hd = 0;
    double f0 = f0seed, w0 = 2.0*M_PI*f0, fnorm = 0.0;
    double *Vr, *Vi, *vsamp, *IRr, *IRi, *Kr, *Ki, *Jr, *Ji, *Fr, *Fi, *Ar, *Ai, *br, *bi;
    struct pac_harm hd;

    if (N <= 0 || K < 1 || oscNode <= 0 || oscNode > N) {
        fprintf(stderr, "hbosc: bad size or oscillator node.\n"); return E_PARMVAL;
    }
    if (Naug > 1600) { fprintf(stderr, "hbosc: system too large (reduce K).\n"); return E_PARMVAL; }

    Vr = TMALLOC(double, Ntot); Vi = TMALLOC(double, Ntot);
    IRr = TMALLOC(double, Ntot); IRi = TMALLOC(double, Ntot);
    Kr = TMALLOC(double, Ntot); Ki = TMALLOC(double, Ntot);
    Fr = TMALLOC(double, Ntot); Fi = TMALLOC(double, Ntot);
    Jr = TMALLOC(double, (size_t)Ntot*(size_t)Ntot); Ji = TMALLOC(double, (size_t)Ntot*(size_t)Ntot);
    Ar = TMALLOC(double, (size_t)Naug*(size_t)Naug); Ai = TMALLOC(double, (size_t)Naug*(size_t)Naug);
    br = TMALLOC(double, Naug); bi = TMALLOC(double, Naug);
    vsamp = TMALLOC(double, (size_t)N*(size_t)P);

    /* seed: fundamental at the osc node (amplitude A -> |V_1| = A/2, real, phase 0) */
    for (i = 0; i < Ntot; i++) { Vr[i] = 0.0; Vi[i] = 0.0; }
    Vr[(size_t)(1+K)*(size_t)N + (size_t)(oscNode-1)] = ampseed * 0.5;
    Vr[(size_t)(K-1)*(size_t)N + (size_t)(oscNode-1)] = ampseed * 0.5;

    for (iter = 0; iter < maxiter; iter++) {
        w0 = 2.0*M_PI*f0;
        for (s = 0; s < P; s++)
            for (i = 0; i < N; i++) {
                double v = 0.0;
                for (k = -K; k <= K; k++) {
                    double ang = 2.0*M_PI*k*s/(double)P;
                    v += Vr[(size_t)(k+K)*(size_t)N+(size_t)i]*cos(ang)
                       - Vi[(size_t)(k+K)*(size_t)N+(size_t)i]*sin(ang);
                }
                vsamp[(size_t)s*(size_t)N+(size_t)i] = v;
            }
        if (hb_extract(ckt, vsamp, N, P, K, &hd, IRr, IRi)) {
            fprintf(stderr, "hbosc: device extraction failed.\n"); rc = E_PARMVAL; break;
        }
        have_hd = 1;
        pac_build_matrix(&hd, f0, 0.0, Jr, Ji, 0);
        {   /* I_C = (J - Jg) V */
            struct pac_harm hg = hd;
            double *Jgr = TMALLOC(double, (size_t)Ntot*(size_t)Ntot);
            double *Jgi = TMALLOC(double, (size_t)Ntot*(size_t)Ntot);
            hg.Cmr = TMALLOC(double, (size_t)hd.nnz*(size_t)(hd.H+1));
            hg.Cmi = TMALLOC(double, (size_t)hd.nnz*(size_t)(hd.H+1));
            pac_build_matrix(&hg, f0, 0.0, Jgr, Jgi, 0);
            FREE(hg.Cmr); FREE(hg.Cmi);
            for (i = 0; i < Ntot; i++) {
                double cr = 0, ci = 0;
                for (c = 0; c < Ntot; c++) {
                    double ar = Jr[(size_t)i*(size_t)Ntot+(size_t)c] - Jgr[(size_t)i*(size_t)Ntot+(size_t)c];
                    double ai = Ji[(size_t)i*(size_t)Ntot+(size_t)c] - Jgi[(size_t)i*(size_t)Ntot+(size_t)c];
                    cr += ar*Vr[c] - ai*Vi[c]; ci += ar*Vi[c] + ai*Vr[c];
                }
                Kr[i] = cr; Ki[i] = ci;
            }
            FREE(Jgr); FREE(Jgi);
        }
        fnorm = 0.0;
        for (i = 0; i < Ntot; i++) {
            Fr[i] = IRr[i] + Kr[i]; Fi[i] = IRi[i] + Ki[i];
            fnorm += Fr[i]*Fr[i] + Fi[i]*Fi[i];
        }
        fnorm = sqrt(fnorm);
        if (verbose)
            fprintf(stderr, "hbosc iter %2d: |F| = %.6e  f0 = %.8g Hz\n", iter, fnorm, f0);
        if (isnan(fnorm) || fnorm > 1e300) { pac_free_harmonics(&hd); have_hd = 0; break; }
        if (fnorm < tol) { rc = OK; break; }        /* converged: keep hd for retention */

        /* bordered Newton: [ H  d ; u*^T 0 ] [dV; dw0] = [-F; 0] */
        memset(Ar, 0, (size_t)Naug*(size_t)Naug*sizeof(double));
        memset(Ai, 0, (size_t)Naug*(size_t)Naug*sizeof(double));
        for (i = 0; i < Ntot; i++)
            for (c = 0; c < Ntot; c++) {
                Ar[(size_t)i*(size_t)Naug+(size_t)c] = Jr[(size_t)i*(size_t)Ntot+(size_t)c];
                Ai[(size_t)i*(size_t)Naug+(size_t)c] = Ji[(size_t)i*(size_t)Ntot+(size_t)c];
            }
        for (i = 0; i < Ntot; i++) {                /* d = dF/dw0 = I_C / w0 */
            Ar[(size_t)i*(size_t)Naug+(size_t)Ntot] = Kr[i]/w0;
            Ai[(size_t)i*(size_t)Naug+(size_t)Ntot] = Ki[i]/w0;
        }
        for (i = 0; i < Ntot; i++) {                /* row = conj(u), u_k = jk V_k */
            int kk = (i/N) - K;
            Ar[(size_t)Ntot*(size_t)Naug+(size_t)i] = -(double)kk * Vi[i];   /* Re conj(u) */
            Ai[(size_t)Ntot*(size_t)Naug+(size_t)i] = -(double)kk * Vr[i];   /* Im conj(u) */
        }
        for (i = 0; i < Ntot; i++) { br[i] = -Fr[i]; bi[i] = -Fi[i]; }
        br[Ntot] = 0.0; bi[Ntot] = 0.0;
        pac_free_harmonics(&hd); have_hd = 0;
        if (pss_csolve(Naug, Ar, Ai, br, bi)) {
            fprintf(stderr, "hbosc: singular augmented system.\n"); rc = E_SINGULAR; break;
        }
        for (i = 0; i < Ntot; i++) { Vr[i] += br[i]; Vi[i] += bi[i]; }
        f0 += br[Ntot] / (2.0*M_PI);                /* dw0 = Re(x[Ntot]) */
    }

    if (rc == OK && have_hd) {
        int numNames, error;
        IFuid *nameList = NULL;
        /* retain the operating point for phasenoise */
        if (osc_valid) { pac_free_harmonics(&osc_hd); FREE(osc_Vr); FREE(osc_Vi); }
        osc_hd = hd;                                /* transfer ownership of hd's arrays */
        osc_Vr = TMALLOC(double, Ntot); osc_Vi = TMALLOC(double, Ntot);
        memcpy(osc_Vr, Vr, (size_t)Ntot*sizeof(double));
        memcpy(osc_Vi, Vi, (size_t)Ntot*sizeof(double));
        osc_f0 = f0; osc_node = oscNode; osc_valid = 1;

        error = CKTnames(ckt, &numNames, &nameList);
        fprintf(stdout, "\nHBOSC: autonomous oscillator steady state\n"
                        "  oscillation frequency f0 = %.9g Hz  (converged, |F| = %.3e)\n"
                        "  node      harmonic   frequency [Hz]        |V|            phase [deg]\n",
                f0, fnorm);
        for (i = 0; i < N; i++) {
            const char *nm = (!error && i < numNames) ? (const char *) nameList[i] : "?";
            for (k = 0; k <= K; k++) {
                double sc = (k == 0) ? 1.0 : 2.0;
                double vr = sc*Vr[(size_t)(k+K)*(size_t)N+(size_t)i];
                double vi = sc*Vi[(size_t)(k+K)*(size_t)N+(size_t)i];
                fprintf(stdout, "  %-8s  %6d   %16.6e   %14.6e   %10.3f\n",
                        nm, k, k*f0, hypot(vr, vi), (k==0)?0.0:atan2(vi, vr)*180.0/M_PI);
            }
        }
        if (nameList) tfree(nameList);
    } else {
        if (have_hd) pac_free_harmonics(&hd);
        if (rc != OK)
            fprintf(stderr, "hbosc: did not converge to an oscillation (try a better fguess/tstab).\n");
    }

    FREE(Vr); FREE(Vi); FREE(IRr); FREE(IRi); FREE(Kr); FREE(Ki);
    FREE(Fr); FREE(Fi); FREE(Jr); FREE(Ji); FREE(Ar); FREE(Ai); FREE(br); FREE(bi); FREE(vsamp);
    return rc;
}

int
PhaseNoiseAnalyze(CKTcircuit *ckt, double fstart, double fstop, int npts, int verbose)
{
    struct pac_harm *hd = &osc_hd;
    int    N, M, Ntot, i, j, k, rc = OK, onode, ni, mi, ei;
    double f0 = osc_f0;
    double *Psr, *Psi, Pcar, freq, mult;
    NOISEAN nj; Ndata data; JOB *oldJob;

    if (!osc_valid) {
        fprintf(stderr, "phasenoise: no oscillator operating point -- run `hbosc` first.\n");
        return E_NOTFOUND;
    }
    N = hd->N; M = hd->M; Ntot = hd->Ntot; onode = osc_node;
    Psr = TMALLOC(double, Ntot); Psi = TMALLOC(double, Ntot);

    /* carrier power at the oscillator node: single-sided amplitude A = 2|V_1|, mean
     * square A^2/2 = 2|V_1|^2 (the reference the sideband noise is measured against). */
    {
        size_t c1 = (size_t)(M+1)*(size_t)N + (size_t)(onode-1);
        Pcar = 2.0 * (osc_Vr[c1]*osc_Vr[c1] + osc_Vi[c1]*osc_Vi[c1]);
        if (Pcar <= 0.0) Pcar = 1.0;
    }

    /* bias the devices at the oscillator op-point (phase-0 sample) for their noise PSD */
    for (j = 1; j <= N; j++) {
        double v = 0.0;
        for (k = -M; k <= M; k++)
            v += osc_Vr[(size_t)(k+M)*(size_t)N+(size_t)(j-1)];
        ckt->CKTrhsOld[j] = v;
    }
    ckt->CKTrhsOld[0] = 0.0;
    ckt->CKTmode = (ckt->CKTmode & MODEUIC) | MODEDCOP | MODEINITSMSIG;
    CKTload(ckt);
    memset(&nj, 0, sizeof(nj));
    nj.NstartFreq = fstart; nj.NstopFreq = fstop; nj.NnumSteps = npts;
    nj.NstpType = 0; nj.NStpsSm = 0; nj.JOBname = "phasenoise";
    memset(&data, 0, sizeof(data)); data.prtSummary = FALSE;
    oldJob = ckt->CKTcurJob; ckt->CKTcurJob = (JOB *) &nj;
    for (i = 0; i < DEVmaxnum; i++)
        if (DEVices[i] && DEVices[i]->DEVnoise && ckt->CKThead[i]) {
            double dummy = 0.0;
            DEVices[i]->DEVnoise(N_DENS, N_OPEN, ckt->CKThead[i], ckt, &data, &dummy);
        }

    fprintf(stdout, "\nPhaseNoise: oscillator phase noise (f0 = %.9g Hz, carrier power %.4e)\n"
                    "  offset [Hz]        L(df) [dBc/Hz]\n", f0, Pcar);
    mult = (npts > 1) ? pow(fstop/fstart, 1.0/(double)(npts-1)) : 1.0;

    for (freq = fstart, i = 0; i < npts; i++, freq *= mult) {
        /* adjoint of H at OFFSET f_in = df, unit at the CARRIER sideband (m=1) of the
         * osc node: Psi is the transimpedance from a noise injection at every (node,
         * sideband) to the carrier. As df -> 0, H(df) -> the singular limit-cycle matrix,
         * so Psi (through the phase mode) blows up as 1/df -> Sv ~ 1/df^2 = phase noise. */
        double *Ar = TMALLOC(double, (size_t)Ntot*(size_t)Ntot);
        double *Ai = TMALLOC(double, (size_t)Ntot*(size_t)Ntot);
        double Sv = 0.0;
        memset(Ar, 0, (size_t)Ntot*(size_t)Ntot*sizeof(double));
        memset(Ai, 0, (size_t)Ntot*(size_t)Ntot*sizeof(double));
        for (ni = 0; ni <= 2*M; ni++)
            for (mi = 0; mi <= 2*M; mi++) {
                int dm = (ni - M) - (mi - M);
                /* small-signal adjoint: ROW sideband frequency (see
                 * pac_build_matrix; RF-audit fix) */
                double omega = 2.0*M_PI*(freq + (double)(ni - M)*f0);
                for (ei = 0; ei < hd->nnz; ei++) {
                    size_t hi = (size_t)ei*(size_t)(hd->H+1) + (size_t)abs(dm);
                    double gr = hd->Gmr[hi], gi = hd->Gmi[hi], cr = hd->Cmr[hi], ci = hd->Cmi[hi];
                    double er, eii; size_t row, col;
                    if (dm < 0) { gi = -gi; ci = -ci; }
                    er = gr - omega*ci; eii = gi + omega*cr;
                    row = (size_t)ni*(size_t)N + (size_t)(hd->rr[ei]-1);
                    col = (size_t)mi*(size_t)N + (size_t)(hd->cc[ei]-1);
                    Ar[col*(size_t)Ntot+row] += er;     /* transpose: [col][row] = H^T */
                    Ai[col*(size_t)Ntot+row] += eii;
                }
            }
        memset(Psr, 0, (size_t)Ntot*sizeof(double));
        memset(Psi, 0, (size_t)Ntot*sizeof(double));
        Psr[(size_t)(M+1)*(size_t)N + (size_t)(onode-1)] = 1.0;   /* carrier sideband m=1 */
        if (pss_csolve(Ntot, Ar, Ai, Psr, Psi) == 0) {
            data.delFreq = 0.0; data.prtSummary = FALSE;
            for (k = -M; k <= M; k++) {
                double dens = 0.0;
                size_t blk = (size_t)(k+M)*(size_t)N;
                /* Enhancement-177: sideband k folds noise ORIGINATING at
                 * |freq + k*f0| -- crucial for the flicker (1/f^3) region,
                 * where evaluating the folded far-sideband blocks at the tiny
                 * offset frequency would wildly overweight them. */
                data.freq = fabs(freq + (double)k * f0);
                if (data.freq == 0.0)
                    data.freq = freq;
                for (j = 1; j <= N; j++) { ckt->CKTrhs[j] = Psr[blk+(size_t)(j-1)]; ckt->CKTirhs[j] = Psi[blk+(size_t)(j-1)]; }
                ckt->CKTrhs[0] = 0.0; ckt->CKTirhs[0] = 0.0;
                for (ei = 0; ei < DEVmaxnum; ei++)
                    if (DEVices[ei] && DEVices[ei]->DEVnoise && ckt->CKThead[ei])
                        DEVices[ei]->DEVnoise(N_DENS, N_CALC, ckt->CKThead[ei], ckt, &data, &dens);
                Sv += dens;
            }
        }
        FREE(Ar); FREE(Ai);
        fprintf(stdout, "  %14.6e   %12.4f\n", freq, 10.0*log10(Sv / Pcar));
        if (npts == 1) break;
    }
    ckt->CKTcurJob = oldJob;
    (void) verbose;

    FREE(Psr); FREE(Psi);
    return rc;
}


int
DCpss(CKTcircuit *ckt,
       int restart)   /* forced restart flag */
{
    PSSan *job = (PSSan *) ckt->CKTcurJob;

    int i;
    double olddelta;
    double delta;
    double newdelta;
    double *temp;
    double startdTime;
    double startsTime;
    double startlTime;
    double startkTime;
    double startTime;
    int startIters;
    int converged;
    int firsttime;
    int error;
    int save_order;
    long save_mode;
    IFuid timeUid;
    IFuid *nameList;
    int numNames;
    double maxstepsize = 0.0;

    int ltra_num;
    CKTnode *node;

    /* New variables */
    int j, oscnNode;
    IFuid freqUid;

    enum {STABILIZATION, SHOOTING, PSS} pss_state = STABILIZATION;

    double err = 0, predsum = 0 ;
    double time_temp = 0, gf_history [HISTORY], rr_history [HISTORY], predsum_history [HISTORY], nextstep ;
    int msize, shooting_cycle_counter = 0;
    double *RHS_copy_se, *RHS_copy_der, *RHS_derivative, *pred, err_0 = HUGE_VAL ;
    double time_err_min_1 = 0, time_err_min_0 = 0, err_min_0 = HUGE_VAL, err_min_1 = 0 ;
    double err_1 = 0, err_max = HUGE_VAL ;
    int pss_points_cycle = 0, dynamic_test = 0 ;
    int pss_driven = 0 ;    /* Enhancement-176: driven (non-autonomous) circuit */
    double gf_last_0 = HUGE_VAL, gf_last_1 = GF_LAST ;
    double thd = 0 ;
    double *psstimes, *pssvalues;
    double *pssstates;   /* Enhancement-119: device states captured per PSS sample */
    double *RHS_max, *RHS_min, *err_conv ;

    /* Francesco Lannutti's MOD */
    /* Stuff needed by frequency estimation reiteration, based on the DFT result */
    int position;
    double max_freq;


    /* Enhancement-117: PSS is a Sparse-1.3-domain analysis. Its shooting loop
     * does not converge under the KLU solver -- it stalls at the first shooting
     * cycle -- while a plain `.tran` on the same circuit runs fine under KLU, so
     * the issue is specific to the periodic breakpoint/timestep machinery, not
     * KLU transients in general. Fail fast with a clear message instead of
     * spinning; `.pss` users simply keep the default Sparse solver. */
    if (ckt->CKTmatrix && ckt->CKTmatrix->CKTkluMODE) {
        fprintf(stderr, "Error: periodic steady state analysis (.pss) is not "
                "supported with 'option KLU'; use 'option sparse' (the default "
                "solver) for .pss.\n");
        return(E_UNSUPP);
    }

    /* Print some useful information */
    PSSDBG( "Periodic Steady State Analysis Started\n\n") ;
    PSSDBG( "PSS Guessed Frequency %g\n", ckt->CKTguessedFreq) ;
    PSSDBG( "PSS Points %ld\n", ckt->CKTpsspoints) ;
    PSSDBG( "PSS Harmonics number %d\n", ckt->CKTharms) ;
    PSSDBG( "PSS Steady Coefficient %g\n", ckt->CKTsteady_coeff) ;
    PSSDBG( "PSS sc_iter %d\n", ckt->CKTsc_iter) ;
    PSSDBG( "PSS Stabilization Time %g\n", ckt->CKTstabTime) ;


    /* Enhancement-176: DRIVEN-mode detection. The Lannutti shooting was built
     * for autonomous oscillators: it hunts the fundamental frequency through a
     * fine forced-breakpoint grid whose spacing is proportional to
     * steady_coeff -- on a driven circuit at steady_coeff = 5e-6 that is a
     * sub-picosecond grid, i.e. millions of forced timesteps per shooting
     * cycle. But when the circuit contains a time-varying independent source
     * (a SIN/PULSE/... V or I source) the period is EXACTLY the source period:
     * no frequency estimation is needed, so the grid (and the estimator) is
     * skipped entirely and each shooting cycle runs at plain-transient speed.
     * The autonomous (oscillator) path is untouched. */
    {
        int vt = CKTtypelook("Vsource") ;
        int itk = CKTtypelook("Isource") ;
        if (vt >= 0 && ckt->CKThead [vt]) {
            VSRCmodel *vm ;
            VSRCinstance *vi ;
            for (vm = (VSRCmodel *) ckt->CKThead [vt] ; vm && !pss_driven ; vm = VSRCnextModel (vm))
                for (vi = VSRCinstances (vm) ; vi ; vi = VSRCnextInstance (vi))
                    if (vi->VSRCfuncTGiven) { pss_driven = 1 ; break ; }
        }
        if (itk >= 0 && ckt->CKThead [itk] && !pss_driven) {
            ISRCmodel *im ;
            ISRCinstance *ii ;
            for (im = (ISRCmodel *) ckt->CKThead [itk] ; im && !pss_driven ; im = ISRCnextModel (im))
                for (ii = ISRCinstances (im) ; ii ; ii = ISRCnextInstance (ii))
                    if (ii->ISRCfuncTGiven) { pss_driven = 1 ; break ; }
        }
        if (pss_driven) {
            /* Match the shooting-phase integration resolution to the sampling
             * resolution the user asked for (psspoints samples per period).
             * Without the (removed) breakpoint flood the LTE control would let
             * steps grow to CKTmaxStep = T/2, and the shooting then converges
             * to the fixed point of that COARSE discretization -- several
             * percent off the true orbit -- which the fine-stepped sampling
             * pass afterwards drifts away from (an inconsistent retained
             * period). One clamp keeps orbit and samples on the same
             * discretization; the cycle count is unchanged, so this is still
             * orders of magnitude faster than the grid flood. */
            double h_samp = (1.0 / ckt->CKTguessedFreq) / (double) ckt->CKTpsspoints ;
            if (ckt->CKTmaxStep > h_samp)
                ckt->CKTmaxStep = h_samp ;
            fprintf (stderr, "PSS: driven circuit detected -- shooting at the fixed source period (%g Hz), frequency estimation off\n",
                     ckt->CKTguessedFreq) ;
        }
    }

    oscnNode = job->PSSoscNode->number ;


    /* Variables and memory initialization */

    for (i = 0 ; i < HISTORY ; i++)
    {
        rr_history [i] = 0.0 ;
        gf_history [i] = 0.0 ;
    }

    msize = SMPmatSize (ckt->CKTmatrix) ;
    RHS_copy_se = TMALLOC (double, msize) ;  /* Set the current RHS reference for next Shooting Evaluation */
    RHS_copy_der = TMALLOC (double, msize) ; /* Used to compute current derivative */
    RHS_derivative = TMALLOC (double, msize) ;
    pred = TMALLOC (double, msize) ;
    RHS_max = TMALLOC (double, msize) ;
    RHS_min = TMALLOC (double, msize) ;
    err_conv = TMALLOC (double, msize) ;
    
    for (i = 0 ; i < msize ; i++)
    {
        RHS_copy_se [i] = 0.0 ;
        RHS_copy_der [i] = 0.0 ;
        RHS_derivative [i] = 0.0 ;
        pred [i] = 0.0 ;
    }

    psstimes = TMALLOC (double, ckt->CKTpsspoints + 1) ;
    pssvalues = TMALLOC (double, msize * (ckt->CKTpsspoints + 1)) ;
    /* Enhancement-119: also capture the device states (charges/fluxes) per
     * sample -- CKTstate0 holds the accepted state alongside CKTrhsOld. */
    pssstates = TMALLOC (double, ckt->CKTnumStates * (ckt->CKTpsspoints + 1)) ;

    for (i = 0 ; i < ckt->CKTpsspoints + 1 ; i++)
        psstimes [i] = 0.0 ;

    for (i = 0 ; i < msize * (ckt->CKTpsspoints + 1) ; i++)
        pssvalues [i] = 0.0 ;

    for (i = 0 ; i < ckt->CKTnumStates * (ckt->CKTpsspoints + 1) ; i++)
        pssstates [i] = 0.0 ;

    /* Delta timestep and circuit time setup */
    delta = ckt->CKTstep ;
    ckt->CKTtime = 0;
    ckt->CKTfinalTime = ckt->CKTstabTime ;

    /* Starting PSS Algorithm, based on Transient Analysis */
    if(restart || ckt->CKTtime == 0) {
        delta = MIN (1 / ckt->CKTguessedFreq / 100, ckt->CKTstep) ;

#ifdef STEPDEBUG
        PSSDBG( "delta = %g    finalTime/200: %g    CKTstep: %g\n", delta, ckt->CKTfinalTime / 200, ckt->CKTstep) ;
#endif
        /* begin LTRA code addition */
        if (ckt->CKTtimePoints != NULL)
            FREE(ckt->CKTtimePoints);

        if (ckt->CKTstep >= ckt->CKTmaxStep)
            maxstepsize = ckt->CKTstep;
        else
            maxstepsize = ckt->CKTmaxStep;

        ckt->CKTsizeIncr = 100;
        ckt->CKTtimeIndex = -1; /* before the DC soln has been stored */
        ckt->CKTtimeListSize = (int)(1 / ckt->CKTguessedFreq / maxstepsize + 0.5);
        ltra_num = CKTtypelook("LTRA");
        if (ltra_num >= 0 && ckt->CKThead[ltra_num] != NULL)
            ckt->CKTtimePoints = TMALLOC(double, ckt->CKTtimeListSize);
        /* end LTRA code addition */

        /* Breakpoints initialization */
        if(ckt->CKTbreaks) FREE(ckt->CKTbreaks);
        ckt->CKTbreaks = TMALLOC(double, 2);
        if(ckt->CKTbreaks == NULL) return(E_NOMEM);
        ckt->CKTbreaks[0] = 0;
        ckt->CKTbreaks[1] = ckt->CKTfinalTime;
        ckt->CKTbreakSize = 2;

#ifdef XSPICE
/* gtri - begin - wbk - 12/19/90 - Modify setting of CKTminBreak */
        /* if (ckt->CKTminBreak == 0)
               ckt->CKTminBreak = ckt->CKTmaxStep * 5e-5 ; */

        /* Set to 10 times delmin for ATESSE 1 compatibity */
        if(ckt->CKTminBreak==0) ckt->CKTminBreak = 10.0 * ckt->CKTdelmin;
/* gtri - end - wbk - 12/19/90 - Modify setting of CKTminBreak */
#else
        /* Minimum Breakpoint Setup */
        if(ckt->CKTminBreak==0) ckt->CKTminBreak=ckt->CKTmaxStep*5e-5;
#endif

#ifdef XSPICE
        /* set anal_init and anal_type */

        /* Tell the code models what mode we're in */
        g_mif_info.circuit.anal_type = MIF_DC;

        g_mif_info.circuit.anal_init = MIF_TRUE;
#endif

	/* Time Domain plot start and prepared to be filled in later */
        error = CKTnames(ckt,&numNames,&nameList);
        if(error) return(error);
        SPfrontEnd->IFnewUid (ckt, &timeUid, NULL, "time", UID_OTHER, NULL);
        error = SPfrontEnd->OUTpBeginPlot (ckt, ckt->CKTcurJob,
                                           "Time Domain Periodic Steady State Analysis",
                                           timeUid, IF_REAL,
                                           numNames, nameList, IF_REAL,
                                           &(job->PSSplot_td));
        tfree(nameList);
        if(error) return(error);

        /* Initialization for Transient Analysis */
        firsttime = 1;
        save_mode = (ckt->CKTmode&MODEUIC) | MODETRANOP | MODEINITJCT;
        save_order = ckt->CKTorder;

#ifdef XSPICE
/* gtri - begin - wbk - set a breakpoint at end of supply ramping time */
        /* must do this after CKTtime set to 0 above */
        if(ckt->enh->ramp.ramptime > 0.0)
            CKTsetBreak(ckt, ckt->enh->ramp.ramptime);
/* gtri - end - wbk - set a breakpoint at end of supply ramping time */

/* gtri - begin - wbk - Call EVTop if event-driven instances exist */
        if(ckt->evt->counts.num_insts != 0) {
            /* use new DCOP algorithm */
            converged = EVTop(ckt,
                        (ckt->CKTmode & MODEUIC) | MODETRANOP | MODEINITJCT,
                        (ckt->CKTmode & MODEUIC) | MODETRANOP | MODEINITFLOAT,
                        ckt->CKTdcMaxIter,
                        MIF_TRUE);
            EVTdump(ckt, IPC_ANAL_DCOP, 0.0);

            EVTop_save(ckt, MIF_FALSE, 0.0);

/* gtri - end - wbk - Call EVTop if event-driven instances exist */
        } else
#endif

        /* Looking for a working Operating Point */
            converged = CKTop(ckt,
                (ckt->CKTmode & MODEUIC) | MODETRANOP | MODEINITJCT,
                (ckt->CKTmode & MODEUIC) | MODETRANOP | MODEINITFLOAT,
                ckt->CKTdcMaxIter);

#if defined(STEPDEBUG) || defined(PSSDEBUG)
        if(converged != 0) {
            fprintf(stdout,"\nTransient solution failed -\n");
            CKTncDump(ckt);
            fprintf(stdout,"\n");
            fflush(stdout);
        } else if (!ft_noacctprint && !ft_noinitprint) {
            fprintf(stdout,"\nInitial Transient Solution\n");
            fprintf(stdout,"--------------------------\n\n");
            fprintf(stdout,"%-30s %15s\n", "Node", "Voltage");
            fprintf(stdout,"%-30s %15s\n", "----", "-------");
            for(node=ckt->CKTnodes->next;node;node=node->next) {
                if (strstr(node->name, "#branch") || !strchr(node->name, '#'))
                    fprintf(stdout,"%-30s %15g\n", node->name,
                                              ckt->CKTrhsOld[node->number]);
            }
            fprintf(stdout,"\n");
            fflush(stdout);
        }
#endif

        /* If no convergence reached - NO valid Operating Point */
        if(converged != 0)
            return(converged);

#ifdef XSPICE
        g_mif_info.circuit.anal_init = MIF_TRUE;

        /* Tell the code models what mode we're in */
        g_mif_info.circuit.anal_type = MIF_TRAN;

        /* Initialize the temporary breakpoint variables to infinity */
        g_mif_info.breakpoint.current = HUGE_VAL;
        g_mif_info.breakpoint.last    = HUGE_VAL;
#endif
        ckt->CKTstat->STATtimePts ++;

        /* Setting Integration Order to Backward Euler */
        ckt->CKTorder = 1;

        /* Copying the maxStep to every deltaOld */
        for(i=0;i<7;i++) {
            ckt->CKTdeltaOld[i]=ckt->CKTmaxStep;
        }

        /* Setting DELTA */
        ckt->CKTdelta = delta;
#ifdef STEPDEBUG
        PSSDBG( "delta initialized to %g\n", ckt->CKTdelta);
#endif

        ckt->CKTsaveDelta = ckt->CKTfinalTime/50;

        ckt->CKTmode = (ckt->CKTmode&MODEUIC) | MODETRAN | MODEINITTRAN;
        /* Changing Circuit MODE */
        /* modeinittran set here */
        ckt->CKTag[0]=ckt->CKTag[1]=0;

        /* State0 copied into State1 - DEPRECATED LEGACY function - to be replaced with memmove() */
        memcpy(ckt->CKTstate1, ckt->CKTstate0,
              (size_t) ckt->CKTnumStates * sizeof(double));

        /* Statistics Initialization using a macro at the beginning of this code */
        INIT_STATS();

    } else {
        /* saj As traninit resets CKTmode */
        ckt->CKTmode = (ckt->CKTmode&MODEUIC) | MODETRAN | MODEINITPRED;
        /* saj */
        INIT_STATS();
        if(ckt->CKTminBreak==0) ckt->CKTminBreak=ckt->CKTmaxStep*5e-5;
        firsttime=0;
        /* To get rawfile working saj*/
        error = SPfrontEnd->OUTpBeginPlot (NULL, NULL,
                                           NULL,
                                           NULL, 0,
                                           666, NULL, 666,
                                           &(job->PSSplot_td));
        if(error) {
            fprintf(stderr, "Error: Couldn't relink rawfile\n");
            return error;
        }
        /* end saj*/

        /* Skip nextTime if it isn't the firsttime! :) */
        goto resume;
    }

/* 650 */
    nextTime:

    /* begin LTRA code addition */
    if (ckt->CKTtimePoints) {
    ckt->CKTtimeIndex++;
        if (ckt->CKTtimeIndex >= ckt->CKTtimeListSize) {
            /* need more space */
            int need;
            if (pss_state == STABILIZATION)
                need = (int) ceil((ckt->CKTstabTime - ckt->CKTtime) / maxstepsize ) ;
            else
                need = (int) ceil((time_temp + 1 / ckt->CKTguessedFreq - ckt->CKTtime) / maxstepsize) ;

            if (need < ckt->CKTsizeIncr)
                need = ckt->CKTsizeIncr;
            ckt->CKTtimeListSize += need;
            ckt->CKTtimePoints = TREALLOC(double, ckt->CKTtimePoints, ckt->CKTtimeListSize);
            ckt->CKTsizeIncr = (int) ceil(1.4 * ckt->CKTsizeIncr);
        }
        ckt->CKTtimePoints[ckt->CKTtimeIndex] = ckt->CKTtime;
    }
    /* end LTRA code addition */

    /* Check for the timepoint acceptance */
    error = CKTaccept(ckt);
    /* check if current breakpoint is outdated; if so, clear */
    if (ckt->CKTtime > ckt->CKTbreaks[0]) CKTclrBreak(ckt);

    /*
 * Breakpoint handling scheme:
 * When a timepoint t is accepted (by CKTaccept), clear all previous
 * breakpoints, because they will never be needed again.
 *
 * t may itself be a breakpoint, or indistinguishably close. DON'T
 * clear t itself; recognise it as a breakpoint and act accordingly
 *
 * if t is not a breakpoint, limit the timestep so that the next
 * breakpoint is not crossed
 */

#ifdef STEPDEBUG
    PSSDBG( "Delta %g accepted at time %g (finaltime: %g)\n", ckt->CKTdelta, ckt->CKTtime, ckt->CKTfinalTime) ;
    fflush(stderr);
#endif /* STEPDEBUG */
    ckt->CKTstat->STATaccepted ++;
    ckt->CKTbreak = 0;
    /* XXX Error will cause single process to bail. */
    if(error)  {
        UPDATE_STATS(DOING_TRAN);
        return(error);
    }
#ifdef XSPICE

    if (wantevtdata) {

        if (pss_state == PSS)
        {
            /* Send event-driven results */
            EVTdump(ckt, IPC_ANAL_TRAN, 0.0);
        }
    } else

#endif

    if (pss_state == PSS)
    {
        nextstep = time_temp + 1 / ckt->CKTguessedFreq * ((double)(pss_points_cycle) / (double)ckt->CKTpsspoints) ;

        /* If in_pss, store data for Time Domain Plot and gather ordered data for FFT computing */
        if ((AlmostEqualUlps (ckt->CKTtime, nextstep, 10)) || (ckt->CKTtime > time_temp + 1 / ckt->CKTguessedFreq))
        {

#ifdef PSSDEBUG
            PSSDBG( "IN_PSS: time point accepted in evolution for FFT calculations.\n") ;
            PSSDBG( "Circuit time %1.15g, final time %1.15g, point index %d and total requested points %ld\n",
                     ckt->CKTtime, nextstep, pss_points_cycle, ckt->CKTpsspoints) ;
#endif

            CKTdump (ckt, ckt->CKTtime, job->PSSplot_td) ;

            /* Store the Time Base for the DFT */
            psstimes [pss_points_cycle] = ckt->CKTtime ;

            /* Store values for the FFT calculation */
            for (i = 1 ; i <= msize ; i++)
                pssvalues [i - 1 + pss_points_cycle * msize] = ckt->CKTrhsOld [i] ;

            /* Enhancement-119: capture the device states at this sample */
            memcpy (pssstates + (size_t)pss_points_cycle * (size_t)ckt->CKTnumStates,
                    ckt->CKTstate0, (size_t)ckt->CKTnumStates * sizeof(double)) ;

            /* Update PSS counter cycle, used to stop the entire algorithm */
            pss_points_cycle++ ;

            /* Set the next BreakPoint for PSS */
            CKTsetBreak (ckt, time_temp + (1 / ckt->CKTguessedFreq) * ((double)pss_points_cycle / (double)ckt->CKTpsspoints)) ;

#ifdef PSSDEBUG
            PSSDBG( "Next breakpoint set in: %1.15g\n", time_temp + 1 / ckt->CKTguessedFreq * ((double)pss_points_cycle / (double)ckt->CKTpsspoints)) ;
#endif

        } else { 
            /* Algo can enter here but should do nothing */

#ifdef PSSDEBUG
            PSSDBG( "IN_PSS: time point accepted in evolution but dropped for FFT calculations\n") ;
#endif

        }
    }

#ifdef XSPICE
/* gtri - begin - wbk - Update event queues/data for accepted timepoint */
    /* Note: this must be done AFTER sending results to SI so it can't */
    /* go next to CKTaccept() above */
    if(ckt->evt->counts.num_insts > 0)
        EVTaccept(ckt, ckt->CKTtime);
/* gtri - end - wbk - Update event queues/data for accepted timepoint */
#endif
    ckt->CKTstat->STAToldIter = ckt->CKTstat->STATnumIter;

    /* ***********************************/
    /* ******* SHOOTING CODE BLOCK *******/
    /* ***********************************/
    switch(pss_state) {

    case STABILIZATION:
    {
        /* Test if stabTime has been reached */
        if (AlmostEqualUlps (ckt->CKTtime, ckt->CKTstabTime, 100))
        {
            time_temp = ckt->CKTtime ;

            /* Set the new Final Time - This is important because the last breakpoint is always CKTfinalTime */
            ckt->CKTfinalTime = time_temp + 2 / ckt->CKTguessedFreq ;
            PSSDBG( "Exiting from stabilization\n") ;
            PSSDBG( "Time of first shooting evaluation will be %1.10g\n", time_temp + 1 / ckt->CKTguessedFreq) ;

            /* Next time is no more in stabilization - Unset the flag */
            pss_state = SHOOTING;

            /* Save the RHS_copy_der as the NEW CKTrhsOld */
            for (i = 1 ; i <= msize ; i++)
                RHS_copy_der [i - 1] = ckt->CKTrhsOld [i] ;
            if (ft_ngdebug) {
                /* Print RHS on exiting from stabilization */
                PSSDBG( "RHS on exiting from stabilization: ");
                for (i = 1; i <= msize; i++)
                {
                    RHS_copy_se[i - 1] = ckt->CKTrhsOld[i];
                    PSSDBG( "%-15g ", RHS_copy_se[i - 1]);
                }
                PSSDBG( "\n");
            }

            /* RHS_max and RHS_min initialization - HUGE_VAL is the maximum machine error */
            for (i = 0 ; i < msize ; i++)
            {
                RHS_max [i] = -HUGE_VAL ;
                RHS_min [i] = HUGE_VAL ;
            }
        }
    }
    break;

    case SHOOTING:
    {
        double offset, interval, nextBreak ;
        /* Calculation of error norms of RHS solution of every accepted nextTime */
        err = 0 ;
        for (i = 0 ; i < msize ; i++)
        {
            /* Save max per node or branch of every estimated period */
            if (RHS_max [i] < ckt->CKTrhsOld [i + 1])
                RHS_max [i] = ckt->CKTrhsOld [i + 1] ;

            /* Save min per node or branch of every estimated period */
            if (RHS_min [i] > ckt->CKTrhsOld [i + 1])
                RHS_min [i] = ckt->CKTrhsOld [i + 1] ;

            /* CKTrhsOld is the last CORRECT value of RHS */
            err_conv [i] = ckt->CKTrhsOld [i + 1] - RHS_copy_se [i] ;
            err += err_conv [i] * err_conv [i] ;

            /* Compute and store derivative */
            RHS_derivative [i] = (ckt->CKTrhsOld [i + 1] - RHS_copy_der [i]) / ckt->CKTdelta ;

            /* Save the RHS_copy_der as the NEW CKTrhsOld */
            RHS_copy_der [i] = ckt->CKTrhsOld [i + 1] ;

#ifdef PSSDEBUG
            PSSDBG( "Pred is so high or so low! Diff is: %g\n", err_conv [i]) ;
#endif

        }
        err = sqrt (err) ;

        /* Start frequency estimation (autonomous mode only) */
        if (!pss_driven &&
            (err < err_0) && (ckt->CKTtime >= time_temp + 0.5 / ckt->CKTguessedFreq)) /* far enough from time temp... */
        {
            if (err < err_min_0)
            {
                err_min_1 = err_min_0 ;            /* store previous minimum of RHS vector error */
                err_min_0 = err ;                  /* store minimum of RHS vector error */
                time_err_min_1 = time_err_min_0 ;  /* store previous minimum of RHS vector error time */
                time_err_min_0 = ckt->CKTtime ;    /* store minimum of RHS vector error time */
            }
        }
        err_0 = err ;

        if ((err > err_1) && (ckt->CKTtime >= time_temp + 0.1 / ckt->CKTguessedFreq)) /* far enough from time temp... */
        {
            if (err > err_max)
                err_max = err ;                /* store maximum of RHS vector error */
        }
        err_1 = err ;


        /* *************************************** */
        /* ********** Breakpoint update ********** */
        /* *************************************** */

        /* Force the tran analysis to evaluate requested breakpoints. Breakpoints are even more closer as
           the next occurence of guessed period is approaching. La lunga notte dei robot viventi... */


        if (pss_driven)
        {
            /* Enhancement-176: the period is the exact source period; no
             * frequency estimation, so no mid-period breakpoint grid is
             * needed -- the cycle-end breakpoint is set by the shooting
             * logic below and the LTE control resolves the waveform. */
        }
        else if ((ckt->CKTtime > time_temp + (1 / ckt->CKTguessedFreq) * 0.995) && (ckt->CKTtime <= time_temp + (1 / ckt->CKTguessedFreq)))
        {
            offset = time_temp + (1 / ckt->CKTguessedFreq) * 0.995 ;
            interval = (1 / ckt->CKTguessedFreq) * (1 - 0.995) * (ckt->CKTsteady_coeff / 10) ;
            i = (int)((ckt->CKTtime - offset) / interval) ;
            nextBreak = offset + (i + 1) * interval ;
            CKTsetBreak (ckt, nextBreak) ;
        }
        else if ((ckt->CKTtime > time_temp + (1 / ckt->CKTguessedFreq) * 0.8) && (ckt->CKTtime <= time_temp + (1 / ckt->CKTguessedFreq) * 0.995))
        {
            offset = time_temp + (1 / ckt->CKTguessedFreq) * 0.8 ;
            interval = (1 / ckt->CKTguessedFreq) * (0.995 - 0.8) * (ckt->CKTsteady_coeff / 5) ;
            i = (int)((ckt->CKTtime - offset) / interval) ;
            nextBreak = offset + (i + 1) * interval ;
            CKTsetBreak (ckt, nextBreak) ;
        }
        else if ((ckt->CKTtime > time_temp + (1 / ckt->CKTguessedFreq) * 0.5) && (ckt->CKTtime <= time_temp + (1 / ckt->CKTguessedFreq) * 0.8))
        {
            offset = time_temp + (1 / ckt->CKTguessedFreq) * 0.5 ;
            interval = (1 / ckt->CKTguessedFreq) * (0.8 - 0.5) * (ckt->CKTsteady_coeff / 3) ;
            i = (int)((ckt->CKTtime - offset) / interval) ;
            nextBreak = offset + (i + 1) * interval ;
            CKTsetBreak (ckt, nextBreak) ;
        }
        else if ((ckt->CKTtime > time_temp + (1 / ckt->CKTguessedFreq) * 0.1) && (ckt->CKTtime <= time_temp + (1 / ckt->CKTguessedFreq) * 0.5))
        {
            offset = time_temp + (1 / ckt->CKTguessedFreq) * 0.1 ;
            interval = (1 / ckt->CKTguessedFreq) * (0.5 - 0.1) * (ckt->CKTsteady_coeff / 2) ;
            i = (int)((ckt->CKTtime - offset) / interval) ;
            nextBreak = offset + (i + 1) * interval ;
            CKTsetBreak (ckt, nextBreak) ;
        }
        else if ((ckt->CKTtime > time_temp) && (ckt->CKTtime <= time_temp + (1 / ckt->CKTguessedFreq) * 0.1))
        {
            offset = time_temp ;
            interval = (1 / ckt->CKTguessedFreq) * (0.1) * (ckt->CKTsteady_coeff) ;
            i = (int)((ckt->CKTtime - offset) / interval) ;
            nextBreak = offset + (i + 1) * interval ;
            CKTsetBreak (ckt, nextBreak) ;
        } else {
            PSSDBG( "Error: Strange behavior\n") ;
            PSSDBG( "    CKTtime: %g\ntime_temp: %g\n\n", ckt->CKTtime, time_temp) ;
        }

        /* *************************************** */
        /* ******* END Breakpoint update ********* */
        /* *************************************** */


        /* If evolution is near shooting... */
        if ((AlmostEqualUlps (ckt->CKTtime, time_temp + 1 / ckt->CKTguessedFreq, 10)) || (ckt->CKTtime > time_temp + 1 / ckt->CKTguessedFreq))
        {
            char* freq = NULL;

            int excessive_err_nodes = 0 ;

            /* Calculation of error norms of RHS solution of every accepted nextTime */
            predsum = 0 ;
            for (i = 0 ; i < msize ; i++)
            {
                /* Pitagora ha sempre ragione!!! :))) */
                /* pred is treated as FREQUENCY to avoid numerical overflow when derivative is close to ZERO */
                if(RHS_derivative[i] == 0) {
                    pred[i] = 0.;
                }
                else {
                    pred[i] = RHS_derivative[i] / err_conv[i];
                }

#ifdef PSSDEBUG
                PSSDBG( "Pred is so high or so low! Diff is: %g\n", err_conv [i]) ;
#endif

                if ((fabs (pred [i]) > ckt->CKTguessedFreq) || (err_conv [i] == 0))
                {
                    if (pred [i] > 0)
                        pred [i] = ckt->CKTguessedFreq ;
                    else
                        pred [i] = -1.* ckt->CKTguessedFreq ;
                }

                predsum += pred [i] ;

#ifdef PSSDEBUG
                PSSDBG( "Predsum in time before to be divided by dynamic_test has value %g\n", 1 / predsum) ;
                PSSDBG( "Current Diff: %g, Derivative: %g, Frequency Projection: %g\n", err_conv [i], RHS_derivative [i], pred [i]) ;
#endif

            }

            /* no error, let's leave shooting */
            if (predsum == 0.) {
                goto shootingexit;
            }

            if (shooting_cycle_counter == 0)
            {
                /* If first time in shooting we tell about it ! */
                PSSDBG( "In shooting...\n") ;
            }

#ifdef PSSDEBUG
            /* For debugging purpose */
            PSSDBG( "\n----------------\n") ;
            PSSDBG( "Shooting cycle iteration number: %3d ||", shooting_cycle_counter) ;

            if (shooting_cycle_counter > 0)
                PSSDBG( " rr: %g || predsum: %g\n", rr_history [shooting_cycle_counter - 1], 1 / predsum) ;
            else
                PSSDBG( " rr: %g || predsum: %g\n", 0.0, 1 / predsum) ;

//            PSSDBG( "Print of dynamically consistent nodes voltages or branches currents:\n") ;
            /* --------------------- */
#endif

            for (i = 0, node = ckt->CKTnodes->next ; node ; i++, node = node->next)
            {
                /* Voltage Node */
                if (!strchr (node->name, '#'))
                {
                    if (fabs (err_conv [i]) > (fabs (RHS_max [i] - RHS_min [i]) * ckt->CKTreltol + ckt->CKTvoltTol) *
                        ckt->CKTtrtol * ckt->CKTsteady_coeff)
                    {
                        excessive_err_nodes++ ;
                    }

                    /* If the dynamic is below 10uV, it's dropped */
                    if (fabs (RHS_max [i] - RHS_min [i]) > 10 * 1e-6)
                    {
                        dynamic_test++ ; /* test on voltage dynamic consistence */
                    }

                /* Current Node */
                } else {
                    if (fabs (err_conv [i]) > (fabs (RHS_max [i] - RHS_min [i]) * ckt->CKTreltol + ckt->CKTabstol) *
                        ckt->CKTtrtol * ckt->CKTsteady_coeff)
                    {
                        excessive_err_nodes++ ;
                    }

                    /* If the dynamic is below 10nA, it's dropped */
                    if (fabs (RHS_max [i] - RHS_min [i]) > 10 * 1e-9)
                    {
                        dynamic_test++ ; /* test on current dynamic consistence */
                    }
                }
            }

            if (dynamic_test == 0)
            {
                /* Test for dynamic existence */
                fprintf(stderr, "Error: No detectable dynamic on voltages nodes or currents branches.\n    PSS analysis aborted\n") ;

                /* Terminates plot in Time Domain and frees the allocated memory */
                SPfrontEnd->OUTendPlot (job->PSSplot_td) ;
                FREE (RHS_copy_se) ;
                FREE (RHS_copy_der) ;
                FREE (RHS_max) ;
                FREE (RHS_min) ;
                FREE (err_conv) ;
                FREE (psstimes) ;
                FREE (pssvalues) ;
                FREE (pssstates) ;
                return (E_PANIC) ; /* error macro in iferrmsg.h */
            }
            else if (!pss_driven && (time_err_min_0 - time_temp) < 0)
            {
                /* Something has gone wrong... */
                fprintf(stderr, "Error: Cannot find a minimum for error vector in estimated period. Try to adjust tstab! PSS analysis aborted\n") ;

                /* Terminates plot in Time Domain and frees the allocated memory */
                SPfrontEnd->OUTendPlot (job->PSSplot_td) ;
                FREE (RHS_copy_se) ;
                FREE (RHS_copy_der) ;
                FREE (RHS_max) ;
                FREE (RHS_min) ;
                FREE (err_conv) ;
                FREE (psstimes) ;
                FREE (pssvalues) ;
                FREE (pssstates) ;
                return (E_PANIC) ; /* to be corrected with definition of new error macro in iferrmsg.h */
            }

//#ifdef STEPDEBUG
//            PSSDBG( "Global Convergence Error reference: %g, Time Projection: %g.\n",
//                     err_conv_ref / dynamic_test, predsum) ;
//#endif

            /* Take the mean value of time prediction trough the dynamic test variable - predsum becomes TIME */
            predsum = 1 / (predsum * dynamic_test);

            /* Store the predsum history as absolute value */
            predsum_history [shooting_cycle_counter] = fabs (predsum) ;

            /***********************************/
            /*** FREQUENCY ESTIMATION UPDATE ***/
            /***********************************/
            if (pss_driven)
            {
                /* Enhancement-176: driven circuit -- the fundamental IS the
                 * source frequency; no estimation. Rank the iteration history
                 * by the period residual instead of the (meaningless here)
                 * period-correction prediction. */
                predsum_history [shooting_cycle_counter] = err ;
            }
            else if ((err_min_0 == err) || (err_min_0 == HUGE_VAL))
            {
                /* Enters here if guessed frequency is higher than the 'real' value */
                ckt->CKTguessedFreq = 1 / (1 / ckt->CKTguessedFreq + fabs (predsum)) ;
                
#ifdef PSSDEBUG
                PSSDBG( "Frequency DOWN: est per %g, err min %g, err min 1 %g, err max %g, err %g\n",
                         time_err_min_0 - time_temp, err_min_0, err_min_1, err_max, err) ;
#endif

            } else {
                /* Enters here if guessed frequency is lower than the 'real' value */
                ckt->CKTguessedFreq = 1 / (time_err_min_0 - time_temp) ;

#ifdef PSSDEBUG
                PSSDBG( "Frequency UP:  est per %g, err min %g, err min 1 %g, err max %g, err %g\n",
                         time_err_min_0 - time_temp, err_min_0, err_min_1, err_max, err) ;
#endif

            }

            /* Temporary variables to store previous occurrence of guessed frequency */
            gf_last_1 = gf_last_0 ;
            gf_last_0 = ckt->CKTguessedFreq ;

            /* Next evaluation of shooting will be updated time (time_temp) summed to updated guessed period */
            time_temp = ckt->CKTtime ;

            /* Store error history */
            rr_history [shooting_cycle_counter] = err ;
            gf_history [shooting_cycle_counter] = ckt->CKTguessedFreq ;
            shooting_cycle_counter++ ;
            if (getenv("PSSTRACE"))
                fprintf(stderr, "[pss] cyc=%-3d t=%.6g gf=%.10g err=%.3e badnodes=%d predsum=%.3e pts=%d rej=%d nit=%d\n",
                        shooting_cycle_counter - 1, ckt->CKTtime, ckt->CKTguessedFreq,
                        err, excessive_err_nodes, predsum,
                        ckt->CKTstat->STATtimePts, ckt->CKTstat->STATrejected,
                        ckt->CKTstat->STATnumIter) ;

            freq = eng(ckt->CKTguessedFreq, 10, TRUE, FALSE);
            PSSDBG( "Updated guessed frequency: %s Hz.\n", freq) ;
            tfree(freq);
            PSSDBG( "Next shooting evaluation time is %1.10g and current time is %1.10g.\n",
                     time_temp + 1 / ckt->CKTguessedFreq, ckt->CKTtime) ;

            /* Restore maximum and minimum error for next search */
            err_min_0 = HUGE_VAL ;
            err_max = -HUGE_VAL ;
            err_0 = HUGE_VAL ;
            err_1 = -HUGE_VAL ;
            dynamic_test = 0 ;

            /* Reset actual RHS reference for next shooting evaluation */
            for (i = 1 ; i <= msize ; i++)
                RHS_copy_se [i - 1] = ckt->CKTrhsOld [i] ;

#ifdef PSSDEBUG
            PSSDBG( "RHS on new shooting cycle: ") ;
            for (i = 0 ; i < msize ; i++)
                PSSDBG( "%-15g ", RHS_copy_se [i]) ;
            PSSDBG( "\n") ;
#endif

            for (i = 0 ; i < msize ; i++)
            {
                /* Reset max and min per node or branch on every shooting cycle */
                RHS_max [i] = -HUGE_VAL ;
                RHS_min [i] = HUGE_VAL ;
            }

            PSSDBG( "----------------\n\n") ;

shootingexit:
            /* Shooting Exit Condition */
            if ((shooting_cycle_counter > ckt->CKTsc_iter) || (excessive_err_nodes == 0))
            {
                int k ;
                double minimum;
                pss_state = PSS ;

#ifdef PSSDEBUG
                PSSDBG( "\nFrequency estimation (FE) and RHS period residual (PR) evolution\n") ;
#endif

                minimum = predsum_history [0] ;
                k = 0 ;
                for (i = 0 ; i < shooting_cycle_counter ; i++)
                {
                    /* Print some statistics */
                    PSSDBG( "%-3d -> FE: %-15.10g || RR: %15.10g", i, gf_history [i], rr_history [i]) ;

                    /* Take the minimum residual iteration */
                    if (minimum > predsum_history [i])
                    {
                        minimum = predsum_history [i] ;
                        k = i ;
                    }
                    PSSDBG( " || predsum/dynamic_test: %15.10g || minimum: %15.10g\n", predsum_history [i], minimum) ;
                }

                if (excessive_err_nodes == 0)  /* SHOOTING has converged  */
                    ckt->CKTguessedFreq = gf_history [shooting_cycle_counter - 1] ;
                else
                    ckt->CKTguessedFreq = gf_history [k] ;

                /* Save the current Time */
                time_temp = ckt->CKTtime ;

                /* Set the new Final Time - This is important because the last breakpoint is always CKTfinalTime */
                ckt->CKTfinalTime = time_temp + 1 / ckt->CKTguessedFreq ;

                /* Dump the first PSS point for the FFT */
                CKTdump (ckt, ckt->CKTtime, job->PSSplot_td) ;
                psstimes [pss_points_cycle] = ckt->CKTtime ;
                for (i = 1 ; i <= msize ; i++)
                    pssvalues [i - 1 + pss_points_cycle * msize] = ckt->CKTrhsOld [i] ;

                /* Enhancement-119: capture the device states at the first sample */
                memcpy (pssstates + (size_t)pss_points_cycle * (size_t)ckt->CKTnumStates,
                        ckt->CKTstate0, (size_t)ckt->CKTnumStates * sizeof(double)) ;

                /* Update the PSS points counter and set the next Breakpoint */
                pss_points_cycle++ ;
                CKTsetBreak (ckt, time_temp + (1 / ckt->CKTguessedFreq) * ((double)pss_points_cycle / (double)ckt->CKTpsspoints)) ;

                freq = eng(ckt->CKTguessedFreq, 10, TRUE, FALSE); /* engineering notation */
                if (excessive_err_nodes == 0)
                    fprintf(stdout, "\nConvergence reached. Final circuit time is %1.10g seconds (iteration n° %d) and predicted fundamental frequency is %s Hz\n", ckt->CKTtime, shooting_cycle_counter - 1, freq) ;
                else
                    fprintf(stdout, "\nConvergence not reached. However the most near convergence iteration has predicted (iteration %d) a fundamental frequency of %s Hz\n", k, freq) ;
                tfree(freq);

#ifdef PSSDEBUG
                PSSDBG( "time_temp %g\n", time_temp) ;
                PSSDBG( "IN_PSS: FIRST time point accepted in evolution for FFT calculations\n") ;
                PSSDBG( "Circuit time %1.15g, final time %1.15g, point index %d and total requested points %ld\n",
                         ckt->CKTtime, time_temp + 1 / ckt->CKTguessedFreq * ((double)pss_points_cycle / (double)ckt->CKTpsspoints),
                         pss_points_cycle, ckt->CKTpsspoints) ;
                PSSDBG( "Next breakpoint set in: %1.15g\n",
                         time_temp + 1 / ckt->CKTguessedFreq * ((double)pss_points_cycle / (double)ckt->CKTpsspoints)) ;
#endif

            } else {
                /* Set the new Final Time - This is important because the last breakpoint is always CKTfinalTime */
                ckt->CKTfinalTime = time_temp + 1 / ckt->CKTguessedFreq ;

                /* Set next the breakpoint */
                CKTsetBreak (ckt, time_temp + 1 / ckt->CKTguessedFreq) ;
            }
        }
    }
    break;

    case PSS:
    {
        /* The algorithm enters here when in_pss is set */

#ifdef PSSDEBUG
        PSSDBG( "ttemp %1.15g, final_time %1.15g, current_time %1.15g\n", time_temp, time_temp + 1 / ckt->CKTguessedFreq, ckt->CKTtime) ;
#endif

        if ((pss_points_cycle == ckt->CKTpsspoints + 1) || (ckt->CKTtime > ckt->CKTfinalTime))
        {
            double *pssfreqs   = TMALLOC (double, ckt->CKTharms);
            double *pssmags    = TMALLOC (double, ckt->CKTharms);
            double *pssphases  = TMALLOC (double, ckt->CKTharms);
            double *pssnmags   = TMALLOC (double, ckt->CKTharms);
            double *pssnphases = TMALLOC (double, ckt->CKTharms);
            double *pssValues  = TMALLOC (double, ckt->CKTpsspoints + 1);
            double *pssResults = TMALLOC (double, msize * ckt->CKTharms);
            /* Enhancement-210: keep the per-harmonic PHASE too, so the frequency-
               domain spectrum can be published as COMPLEX (mag + phase both
               accessible via mag()/vp(), like the hb command's E-209 vectors and
               like AC-analysis node vectors), instead of magnitude-only. */
            double *pssPhaseR  = TMALLOC (double, msize * ckt->CKTharms);

            /* End plot in Time Domain */
            SPfrontEnd->OUTendPlot (job->PSSplot_td) ;

            /* Frequency Plot Creation */
            error = CKTnames (ckt, &numNames, &nameList) ;
            if (error)
                return (error) ;
            SPfrontEnd->IFnewUid (ckt, &freqUid, NULL, "frequency", UID_OTHER, NULL) ;
            error = SPfrontEnd->OUTpBeginPlot (ckt, ckt->CKTcurJob,
                                               "Frequency Domain Periodic Steady State Analysis",
                                               freqUid, IF_REAL,
                                               numNames, nameList, IF_COMPLEX,
                                               &(job->PSSplot_fd)) ;
            tfree (nameList) ;
            SPfrontEnd->OUTattributes (job->PSSplot_fd, NULL, PLOT_ARROWS, NULL) ;

            /* ******************** */
            /* Starting DFT on data */
            /* ******************** */
            for (i = 0 ; i < msize ; i++)
            {
                for (j = 0 ; j < ckt->CKTpsspoints ; j++)
                    pssValues [j] = pssvalues [j * msize + i] ;

                DFT (ckt->CKTpsspoints, ckt->CKTharms, &thd, psstimes, pssValues, ckt->CKTguessedFreq,
                         pssfreqs, pssmags, pssphases, pssnmags, pssnphases) ;

                for (j = 0 ; j < ckt->CKTharms ; j++)
                {
                    pssResults [j * msize + i] = pssmags [j] ;      /* single-sided magnitude */
                    pssPhaseR  [j * msize + i] = pssphases [j] ;    /* phase, degrees (0 at DC) */
                }
            }

            /* Enhancement-210: publish each harmonic as a COMPLEX data row
               (mag * e^{j*phase}); |.| reproduces the magnitude spectrum and the
               phase is now recoverable via vp(). (DFT returns the phase in degrees.) */
            for (j = 0 ; j < ckt->CKTharms ; j++)
            {
                IFvalue freqData, valData ;
                IFcomplex *cdata = TMALLOC (IFcomplex, msize) ;
                freqData.rValue = pssfreqs [j] ;
                valData.v.numValue = msize ;
                valData.v.vec.cVec = cdata ;
                for (i = 0 ; i < msize ; i++)
                {
                    double m  = pssResults [j * msize + i] ;
                    double ph = pssPhaseR  [j * msize + i] * M_PI / 180.0 ;
                    cdata [i].real = m * cos (ph) ;
                    cdata [i].imag = (j == 0) ? 0.0 : m * sin (ph) ;
                }
                SPfrontEnd->OUTpData (job->PSSplot_fd, &freqData, &valData) ;
                FREE (cdata) ;
            }
            /* ****************** */
            /* Ending DFT on data */
            /* ****************** */

            /* Terminates plot in Frequency Domain and frees the allocated memory */
            SPfrontEnd->OUTendPlot (job->PSSplot_fd) ;

            /* Verify the frequency found */
            max_freq = pssResults [msize] ;             /* max_freq = pssResults [1 * msize + 0] ; */
            position = 1 ;
            for (j = 1 ; j < ckt->CKTharms ; j++)
            {
                for (i = 0 ; i < msize ; i++)
                {
                    if (max_freq < pssResults [j * msize + i])
                    {
                        max_freq = pssResults [j * msize + i] ;
                        position = j ;
                    }
                }
            }

            if (!pss_driven && pssfreqs [position] != ckt->CKTguessedFreq)
            {
                /* autonomous only: for a driven circuit the fundamental IS the
                 * source frequency -- relaunching at the strongest spectral
                 * line (e.g. a rectifier's dominant 2nd harmonic) would retain
                 * the wrong period (Enhancement-176). */
                ckt->CKTguessedFreq = pssfreqs [position] ;
                fprintf(stdout, "\nThe predicted fundamental frequency is incorrect.\nRelaunching the analysis ") ;
                fprintf(stdout, "with new guessed fundamental frequency %.6g Hz\n\n", ckt->CKTguessedFreq) ;
                DCpss (ckt, 1) ;
                /* the relaunched run retained its own (correct) operating point;
                 * this run's samples are stale -- fall through and free them. */
            }
            else
            {
                /* Enhancement-119: frequency confirmed -- retain this converged
                 * periodic operating point on the job for periodic small-signal
                 * reuse (PAC/pnoise/PXF). Ownership of the sample arrays is
                 * transferred to the job (set local ptrs NULL so the FREE below
                 * is a no-op); the DFT that produced the harmonic output above
                 * was taken from exactly these samples, so they are self-consistent. */
                FREE (job->PSSopTimes) ;
                FREE (job->PSSopVoltages) ;
                FREE (job->PSSopStates) ;
                job->PSSopPoints    = ckt->CKTpsspoints ;
                job->PSSopMsize     = msize ;
                job->PSSopNumStates = ckt->CKTnumStates ;
                job->PSSopFreq      = ckt->CKTguessedFreq ;
                job->PSSopTimes     = psstimes ;   psstimes  = NULL ;
                job->PSSopVoltages  = pssvalues ;  pssvalues = NULL ;
                job->PSSopStates    = pssstates ;  pssstates = NULL ;
                fprintf (stderr, "PSS periodic operating point retained: %ld samples x "
                                 "%d unknowns x %d states at f = %.10g Hz\n",
                         job->PSSopPoints, job->PSSopMsize, job->PSSopNumStates,
                         job->PSSopFreq) ;

                /* Self-check: report the osc-node voltage swing straight from the
                 * retained samples, so the retained data can be validated without
                 * a consumer yet (a periodic node must swing over the period). */
                {
                    int onode = job->PSSoscNode ? job->PSSoscNode->number : 0 ;
                    if (onode > 0 && onode <= job->PSSopMsize) {
                        double vmin = HUGE_VAL, vmax = -HUGE_VAL ;
                        long s ;
                        for (s = 0 ; s < job->PSSopPoints ; s++) {
                            double v = job->PSSopVoltages [(onode - 1) + s * job->PSSopMsize] ;
                            if (v < vmin) vmin = v ;
                            if (v > vmax) vmax = v ;
                        }
                        fprintf (stderr, "  retained op-point self-check: osc-node swing "
                                         "[%.6g, %.6g] over the period\n", vmin, vmax) ;
                    }
                }

                /* Enhancement-120: report the periodic small-signal Jacobian
                 * harmonics at the osc node, built from the retained op-point. */
                pss_jacobian_report (ckt, job) ;

                /* Enhancement-121: assemble the harmonic conversion matrix from
                 * the full periodic Jacobian and solve it -- the PAC engine. */
                pss_pac_report (ckt, job) ;

                /* Enhancement-122: for a .pac card, sweep the input frequency and
                 * emit the sideband-0 node responses as a complex plot. */
                if (job->PSSdoPAC)
                    pac_sweep (ckt, job) ;

                /* Enhancement-124: for a .pnoise card, fold each device's noise
                 * through the conversion matrix to get the output noise spectrum. */
                if (job->PSSdoPnoise)
                    pnoise_sweep (ckt, job) ;

                /* Enhancement-125: for a .pxf card, solve the conversion adjoint and
                 * report the input->output transfer at each sideband. */
                if (job->PSSdoPXF)
                    pxf_sweep (ckt, job) ;

#ifdef RFSPICE
                /* Enhancement-132: for a .psp card, inject at each RF port through
                 * the conversion matrix and emit the periodic S-parameters. */
                if (job->PSSdoPSP)
                    psp_sweep (ckt, job) ;
#endif
            }
            /****************************/


            FREE (pssResults) ;
            FREE (pssPhaseR) ;
            FREE (pssValues) ;
            FREE (pssnphases) ;
            FREE (pssnmags) ;
            FREE (pssphases) ;
            FREE (pssmags) ;
            FREE (pssfreqs) ;

            FREE (RHS_copy_se) ;
            FREE (RHS_copy_der) ;
            FREE (RHS_max) ;
            FREE (RHS_min) ;
            FREE (err_conv) ;
            FREE (psstimes) ;
            FREE (pssvalues) ;
            FREE (pssstates) ;
            ckt->CKTag[0] = ckt->CKTag[1] = 0.;
            return (OK) ;
        }
    }
    break;

    } /* switch(pss_state) */

    /* ********************************** */
    /* **** END SHOOTING CODE BLOCK ***** */
    /* ********************************** */

    if(SPfrontEnd->IFpauseTest()) {
        /* user requested pause... */
        UPDATE_STATS(DOING_TRAN);
        return(E_PAUSE);
    }
    /* RESUME */
resume:
#ifdef STEPDEBUG
    if( (ckt->CKTdelta <= ckt->CKTfinalTime/50) &&
        (ckt->CKTdelta <= ckt->CKTmaxStep)) {
        ;
    } else {
        if(ckt->CKTfinalTime/50<ckt->CKTmaxStep) {
	    PSSDBG( "limited by Tstop/50\n");
        } else {
	    PSSDBG( "limited by Tmax == %g\n", ckt->CKTmaxStep);
        }
    }
#endif
#ifdef HAS_PROGREP
    if (ckt->CKTtime == 0.)
        SetAnalyse( "ptran init", 0);
    else if ((pss_state != PSS) && (shooting_cycle_counter > 0))
        SetAnalyse("shooting", shooting_cycle_counter) ;
    else
        SetAnalyse( "ptran", (int)((ckt->CKTtime * 1000.) / ckt->CKTfinalTime));
#endif
    ckt->CKTdelta =
            MIN(ckt->CKTdelta,ckt->CKTmaxStep);
#ifdef XSPICE
/* gtri - begin - wbk - Cut integration order if first timepoint after breakpoint */
    /* if(ckt->CKTtime == g_mif_info.breakpoint.last) */
    if ( AlmostEqualUlps( ckt->CKTtime, g_mif_info.breakpoint.last, 100 ) )
        ckt->CKTorder = 1;
/* gtri - end   - wbk - Cut integration order if first timepoint after breakpoint */

#endif

  /* are we at a breakpoint, or indistinguishably close? */
    /* if ((ckt->CKTtime == ckt->CKTbreaks[0]) || (ckt->CKTbreaks[0] - */
    if (ckt->CKTbreaks [0] - ckt->CKTtime <= ckt->CKTdelmin)
    {
        /*if ( AlmostEqualUlps( ckt->CKTtime, ckt->CKTbreaks[0], 100 ) || (ckt->CKTbreaks[0] -
        *    (ckt->CKTtime) <= ckt->CKTdelmin)) {*/
        /* first timepoint after a breakpoint - cut integration order */
        /* and limit timestep to .1 times minimum of time to next breakpoint,
         * and previous timestep
         */
        ckt->CKTorder = 1;
#ifdef STEPDEBUG
        if( (ckt->CKTdelta > .1*ckt->CKTsaveDelta) ||
            (ckt->CKTdelta > .1*(ckt->CKTbreaks[1] - ckt->CKTbreaks[0])) ) {
            if(ckt->CKTsaveDelta < (ckt->CKTbreaks[1] - ckt->CKTbreaks[0]))  {
                PSSDBG( "limited by pre-breakpoint delta (saveDelta: %1.10g, nxt_breakpt: %1.10g, curr_breakpt: %1.10g and CKTtime: %1.10g\n",
                         ckt->CKTsaveDelta, ckt->CKTbreaks [1], ckt->CKTbreaks [0], ckt->CKTtime) ;
            } else {
                PSSDBG( "limited by next breakpoint\n") ;
                PSSDBG( "(saveDelta: %1.10g, Delta: %1.10g, CKTtime: %1.10g and delmin: %1.10g\n",
                         ckt->CKTsaveDelta, ckt->CKTdelta, ckt->CKTtime, ckt->CKTdelmin) ;
	    }
	}
#endif

        if (ckt->CKTbreaks[1] - ckt->CKTbreaks[0] == 0) {
//            ckt->CKTdelta = ckt->CKTdelmin;
        }
        else
            ckt->CKTdelta = MIN (ckt->CKTdelta, .1 * MIN (ckt->CKTsaveDelta,
            ckt->CKTbreaks[1] - ckt->CKTbreaks[0]));

        if(firsttime) {
            /* set a breakpoint to reduce ringing of current in devices */
            if (ckt->CKTmode & MODEUIC)
                CKTsetBreak(ckt, ckt->CKTstep);
            ckt->CKTdelta /= 10;
#ifdef STEPDEBUG
            PSSDBG( "delta cut for initial timepoint\n");
#endif
        }

#ifndef XSPICE
        /* don't want to get below delmin for no reason */
        ckt->CKTdelta = MAX(ckt->CKTdelta, ckt->CKTdelmin*2.0);
#endif

    }

#ifndef XSPICE
    else if(ckt->CKTtime + ckt->CKTdelta >= ckt->CKTbreaks[0]) {
        ckt->CKTsaveDelta = ckt->CKTdelta;
        ckt->CKTdelta = ckt->CKTbreaks[0] - ckt->CKTtime;
        /* PSSDBG( "delta cut to %g to hit breakpoint\n" ,ckt->CKTdelta) ; */
        fflush(stderr);
        ckt->CKTbreak = 1; /* why? the current pt. is not a bkpt. */
    }
     /* Try to equalise the last two time steps before the breakpoint,
        if the second step would be smaller than CKTdelta otherwise.*/
    else if (ckt->CKTtime + 1.9 * ckt->CKTdelta > ckt->CKTbreaks[0]) {
        ckt->CKTsaveDelta = ckt->CKTdelta;
        ckt->CKTdelta = (ckt->CKTbreaks[0] - ckt->CKTtime) / 2.;
#ifdef STEPDEBUG
        PSSDBG( "Delta equalising step at time %e with delta %e\n", ckt->CKTtime, ckt->CKTdelta);
#endif
    }
#endif /* !XSPICE */


#ifdef XSPICE
/* gtri - begin - wbk - Add Breakpoint stuff */

    if(ckt->CKTtime + ckt->CKTdelta >= g_mif_info.breakpoint.current) {
        /* If next time > temporary breakpoint, force it to the breakpoint */
        /* And mark that timestep was set by temporary breakpoint */
        ckt->CKTsaveDelta = ckt->CKTdelta;
        ckt->CKTdelta = g_mif_info.breakpoint.current - ckt->CKTtime;
        g_mif_info.breakpoint.last = ckt->CKTtime + ckt->CKTdelta;
    } else {
        /* Else, mark that timestep was not set by temporary breakpoint */
        g_mif_info.breakpoint.last = HUGE_VAL;
    }

/* gtri - end - wbk - Add Breakpoint stuff */

/* gtri - begin - wbk - Modify Breakpoint stuff */
    /* Throw out any permanent breakpoint times <= current time */
    while ((ckt->CKTbreaks[0] <= ckt->CKTtime + ckt->CKTminBreak ||
        AlmostEqualUlps(ckt->CKTbreaks[0], ckt->CKTtime, 100)) &&
        ckt->CKTbreaks[0] < ckt->CKTfinalTime) {
#ifdef STEPDEBUG
        PSSDBG("throwing out permanent breakpoint times <= current time "
            "(brk pt: %g)\n",
            ckt->CKTbreaks[0]);
        PSSDBG("    ckt_time: %g    ckt_min_break: %g\n",
            ckt->CKTtime, ckt->CKTminBreak);
#endif
        CKTclrBreak(ckt);
    }
    /* Force the breakpoint if appropriate */
    if(ckt->CKTtime + ckt->CKTdelta > ckt->CKTbreaks[0]) {
        ckt->CKTbreak = 1;
        ckt->CKTsaveDelta = ckt->CKTdelta;
        ckt->CKTdelta = ckt->CKTbreaks[0] - ckt->CKTtime;
    }
    /* Try to equalise the last two time steps before the breakpoint,
       if the second step would be smaller than CKTdelta otherwise.*/
    else if (!AlmostEqualUlps(ckt->CKTtime + ckt->CKTdelta, ckt->CKTfinalTime, 100)
        && ckt->CKTtime + 1.9 * ckt->CKTdelta > ckt->CKTbreaks[0]) {
        ckt->CKTsaveDelta = ckt->CKTdelta;
        ckt->CKTdelta = (ckt->CKTbreaks[0] - ckt->CKTtime) / 2.0;
        #ifdef STEPDEBUG
            PSSDBG( "Delta equalising step at time %e with delta %e\n", ckt->CKTtime, ckt->CKTdelta);
        #endif
    }

/* gtri - end - wbk - Modify Breakpoint stuff */

/* gtri - begin - wbk - Do event solution */

    if(ckt->evt->counts.num_insts > 0) {

        /* if time = 0 and op_alternate was specified as false during */
        /* dcop analysis, call any changed instances to let them */
        /* post their outputs with their associated delays */
        if((ckt->CKTtime == 0.0) && (! ckt->evt->options.op_alternate))
            EVTiter(ckt);

        /* while there are events on the queue with event time <= next */
        /* projected analog time, process them */
        while((g_mif_info.circuit.evt_step = EVTnext_time(ckt))
               <= (ckt->CKTtime + ckt->CKTdelta)) {

            /* Initialize temp analog bkpt to infinity */
            g_mif_info.breakpoint.current = HUGE_VAL;

            /* Pull items off queue and process them */
            EVTdequeue(ckt, g_mif_info.circuit.evt_step);
            EVTiter(ckt);

            /* If any instances have forced an earlier */
            /* next analog time, cut the delta */
            if(ckt->CKTbreaks[0] < g_mif_info.breakpoint.current)
                if(ckt->CKTbreaks[0] > ckt->CKTtime + ckt->CKTminBreak)
                    g_mif_info.breakpoint.current = ckt->CKTbreaks[0];
            if(g_mif_info.breakpoint.current < ckt->CKTtime + ckt->CKTdelta) {
                /* Breakpoint must be > last accepted timepoint */
                /* and >= current event time */
                if(g_mif_info.breakpoint.current >  ckt->CKTtime + ckt->CKTminBreak  &&
                   g_mif_info.breakpoint.current >= g_mif_info.circuit.evt_step) {
                    ckt->CKTsaveDelta = ckt->CKTdelta;
                    ckt->CKTdelta = g_mif_info.breakpoint.current - ckt->CKTtime;
                    g_mif_info.breakpoint.last = ckt->CKTtime + ckt->CKTdelta;
                }
            }

        } /* end while next event time <= next analog time */
    } /* end if there are event instances */

/* gtri - end - wbk - Do event solution */

#endif

    /* What is that??? */
    for(i=5; i>=0; i--)
        ckt->CKTdeltaOld[i+1] = ckt->CKTdeltaOld[i];
    ckt->CKTdeltaOld[0] = ckt->CKTdelta;

    temp = ckt->CKTstates[ckt->CKTmaxOrder+1];
    for(i=ckt->CKTmaxOrder;i>=0;i--) {
        ckt->CKTstates[i+1] = ckt->CKTstates[i];
    }
    ckt->CKTstates[0] = temp;

/* 600 */
    for (;;) {
#ifdef XSPICE
/* gtri - add - wbk - 4/17/91 - Fix Berkeley bug */
/* This is needed here to allow CAPask to output currents */
/* during Transient analysis.  A grep for CKTcurrentAnalysis */
/* indicates that it should not hurt anything else ... */

        ckt->CKTcurrentAnalysis = DOING_TRAN;

/* gtri - end - wbk - 4/17/91 - Fix Berkeley bug */
#endif
        olddelta=ckt->CKTdelta;
        /* time abort? */
        ckt->CKTtime += ckt->CKTdelta;
        ckt->CKTdeltaOld[0]=ckt->CKTdelta;
        NIcomCof(ckt);
#ifdef PREDICTOR
        error = NIpred(ckt);
#endif /* PREDICTOR */
        save_mode = ckt->CKTmode;
        save_order = ckt->CKTorder;
#ifdef XSPICE
/* gtri - begin - wbk - Add Breakpoint stuff */

        /* Initialize temporary breakpoint to infinity */
        g_mif_info.breakpoint.current = HUGE_VAL;

/* gtri - end - wbk - Add Breakpoint stuff */


/* gtri - begin - wbk - add convergence problem reporting flags */
        /* delta is forced to equal delmin on last attempt near line 650 */
        if(ckt->CKTdelta <= ckt->CKTdelmin)
            ckt->enh->conv_debug.last_NIiter_call = MIF_TRUE;
        else
            ckt->enh->conv_debug.last_NIiter_call = MIF_FALSE;
/* gtri - begin - wbk - add convergence problem reporting flags */


/* gtri - begin - wbk - Call all hybrids */

/* gtri - begin - wbk - Set evt_step */

        if(ckt->evt->counts.num_insts > 0) {
            g_mif_info.circuit.evt_step = ckt->CKTtime;
        }
/* gtri - end - wbk - Set evt_step */
#endif

        /* Enhancement-118: under KLU, force a full factorization every PSS
         * timestep. NIiter otherwise reuses `klu_refactor` (the previous pivot
         * ordering, values only) for speed; across PSS's very fine,
         * breakpoint-dense shooting timesteps that reused ordering accumulates
         * just enough numerical error to inflate the local truncation error,
         * which collapses the step size into a ~20-million-step run that never
         * finishes. Setting NISHOULDREORDER makes NIiter take the accurate
         * SMPreorder (`klu_factor`, re-pivoted) path, so KLU PSS converges to
         * the same result as Sparse. (Sparse's own factor is accurate enough to
         * not need this; a plain `.tran` under KLU is unaffected -- its steps
         * are coarse enough that the refactor error never matters.) */
        if (ckt->CKTmatrix->CKTkluMODE)
            ckt->CKTniState |= NISHOULDREORDER;

        converged = NIiter(ckt,ckt->CKTtranMaxIter);

#ifdef XSPICE
        if(ckt->evt->counts.num_insts > 0) {
            g_mif_info.circuit.evt_step = ckt->CKTtime;
            EVTcall_hybrids(ckt);
        }
/* gtri - end - wbk - Call all hybrids */

#endif
        ckt->CKTstat->STATtimePts ++;
        ckt->CKTmode = (ckt->CKTmode&MODEUIC)|MODETRAN | MODEINITPRED;
        if(firsttime) {
            memcpy(ckt->CKTstate2, ckt->CKTstate1,
                   (size_t) ckt->CKTnumStates * sizeof(double));
            memcpy(ckt->CKTstate3, ckt->CKTstate1,
                   (size_t) ckt->CKTnumStates * sizeof(double));
        }
        /* txl, cpl addition */
        if (converged == 1111) {
                return(converged);
        }

#ifdef PSSDEBUG
        if (pss_state == PSS)
            PSSDBG( "pss_state: %d, converged: %d\n", pss_state, converged) ;
#endif
        if(converged != 0) {
            ckt->CKTtime = ckt->CKTtime - ckt->CKTdelta;
            ckt->CKTstat->STATrejected++;
            ckt->CKTdelta = ckt->CKTdelta/8;
#ifdef STEPDEBUG
            PSSDBG( "delta cut to %g for non-convergence\n", ckt->CKTdelta) ;
            fflush(stderr);
#endif
            if(firsttime) {
                ckt->CKTmode = (ckt->CKTmode&MODEUIC) | MODETRAN | MODEINITTRAN;
            }
            ckt->CKTorder = 1;

#ifdef XSPICE
/* gtri - begin - wbk - Add Breakpoint stuff */

        /* Force backup if temporary breakpoint is < current time */
        } else if(g_mif_info.breakpoint.current < ckt->CKTtime) {
            ckt->CKTsaveDelta = ckt->CKTdelta;
            ckt->CKTtime -= ckt->CKTdelta;
            ckt->CKTdelta = g_mif_info.breakpoint.current - ckt->CKTtime;
            g_mif_info.breakpoint.last = ckt->CKTtime + ckt->CKTdelta;

            if(firsttime) {
                ckt->CKTmode = (ckt->CKTmode&MODEUIC)|MODETRAN | MODEINITTRAN;
            }
            ckt->CKTorder = 1;

/* gtri - end - wbk - Add Breakpoint stuff */
#endif

        } else {
            if (firsttime) {
                firsttime = 0;
                goto nextTime;  /* no check on
                                 * first time point
                                 */
            }
            newdelta = ckt->CKTdelta;
            error = CKTtrunc(ckt,&newdelta);
            if(error) {
                UPDATE_STATS(DOING_TRAN);
                return(error);
            }
            if (newdelta > .9 * ckt->CKTdelta) {
                if ((ckt->CKTorder == 1) && (ckt->CKTmaxOrder > 1)) { /* don't rise the order for backward Euler */
                    newdelta = ckt->CKTdelta;
                    ckt->CKTorder = 2;
                    error = CKTtrunc(ckt, &newdelta);
                    if (error) {
                        UPDATE_STATS(DOING_TRAN);
                        return(error);
                    }
                    if (newdelta <= 1.05 * ckt->CKTdelta) {
                        ckt->CKTorder = 1;
                    }
                }
                /* time point OK  - 630 */
                ckt->CKTdelta = newdelta;

#ifdef STEPDEBUG
                PSSDBG( "delta set to truncation error result: %g. Point accepted at CKTtime: %g\n", ckt->CKTdelta, ckt->CKTtime) ;
                fflush(stderr);
#endif


                /* go to 650 - trapezoidal */
                goto nextTime;

            } else { /* newdelta <= .9 * ckt->CKTdelta */
                ckt->CKTtime = ckt->CKTtime -ckt->CKTdelta;
                ckt->CKTstat->STATrejected ++;
                ckt->CKTdelta = newdelta;
#ifdef STEPDEBUG
                PSSDBG( "delta set to truncation error result:point rejected\n") ;
#endif
            }
        }

        if (ckt->CKTdelta <= ckt->CKTdelmin) {
            if (olddelta > ckt->CKTdelmin) {
                ckt->CKTdelta = ckt->CKTdelmin;
#ifdef STEPDEBUG
                PSSDBG( "delta at delmin\n");
#endif
            } else {
                UPDATE_STATS(DOING_TRAN);
                errMsg = CKTtrouble(ckt, "Timestep too small");
                return(E_TIMESTEP);
            }
        }
#ifdef XSPICE
/* gtri - begin - wbk - Do event backup */

        if(ckt->evt->counts.num_insts > 0)
            EVTbackup(ckt, ckt->CKTtime + ckt->CKTdelta);

/* gtri - end - wbk - Do event backup */
#endif
    }
    /* NOTREACHED */
}

static int
DFT
(
    long int ndata,  /* number of entries in the Time and Value arrays */
    int numFreq,     /* number of harmonics to calculate */
    double *thd,     /* total harmonic distortion (percent) to be returned */
    double *Time,    /* times at which the voltage/current values were measured */
    double *Value,   /* voltage or current vector whose transform is desired */
    double FundFreq, /* the fundamental frequency of the analysis */
    double *Freq,    /* the frequency value of the various harmonics */
    double *Mag,     /* the Magnitude of the fourier transform */
    double *Phase,   /* the Phase of the fourier transform */
    double *nMag,    /* the normalized magnitude of the transform: nMag (fund) = 1 */
    double *nPhase   /* the normalized phase of the transform: Nphase (fund) = 0 */
)
{
    /* Note: we can consider these as a set of arrays.  The sizes are:
     * Time [ndata], Value [ndata], Freq [numFreq], Mag [numfreq],
     * Phase [numfreq], nMag [numfreq], nPhase [numfreq]
     *
     * The arrays must all be allocated by the caller.
     * The Time and Value array must be reasonably distributed over at
     * least one full period of the fundamental Frequency for the
     * fourier transform to be useful.  The function will take the
     * last period of the frequency as data for the transform.
     *
     * We are assuming that the caller has provided exactly one period
     * of the fundamental frequency.  */
    int i, j;
    double tmp;

    NG_IGNORE (Time);

    /* clear output/computation arrays */

    for (i = 0; i < numFreq; i++) {
        Mag [i] = 0;
        Phase [i] = 0;
    }

    for (i = 0; i < ndata; i++) {
        for (j = 0; j < numFreq; j++) {
            Mag [j] += (Value [i] * sin (j * 2.0 * M_PI * i / ((double)ndata)));
            Phase [j] += (Value [i] * cos (j * 2.0 * M_PI * i / ((double)ndata)));
        }
    }

    Mag [0] = Phase [0] / (double)ndata;
    Phase [0] = 0;
    nMag [0] = 0;
    nPhase [0] = 0;
    Freq [0] = 0;
    *thd = 0;

    for (i = 1; i < numFreq; i++) {
        tmp = Mag [i] * 2.0 / (double)ndata;
        Phase [i] *= 2.0 / (double)ndata;
        Freq [i] = i * FundFreq;
        Mag [i] = hypot (tmp, Phase [i]);
        Phase [i] = atan2 (Phase [i], tmp) * 180.0 / M_PI;
        nMag [i] = Mag [i] / Mag [1];
        nPhase [i] = Phase [i] - Phase [1];
        if (i > 1)
            *thd += nMag [i] * nMag [i];
    }

    *thd = 100 * sqrt (*thd);
    return (OK);
}
