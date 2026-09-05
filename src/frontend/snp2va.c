/* Enhancement-200: Touchstone (.sNp) -> Verilog-A n-port converter, in C.
 *
 * A C port of the pure-Python snp2va.py (Enhancement-199): parse a Touchstone
 * S-parameter file, convert S -> Y, fit every Y_ij(f) with a common-pole rational
 * (Gustavsen vector fitting), and emit a Verilog-A n-port realized with laplace_nd,
 * so through OpenVAF/OSDI it works in AC and transient. Used by the `pre_snp`
 * front-end command, which then invokes openvaf-r to compile the emitted .va.
 *
 * The numerical core is self-contained (only stdio/stdlib/string/math/complex),
 * so it does not pull in ngspice's own complex.h. Public entry point:
 *     int snp2va_convert(const char *snp, const char *va, const char *module,
 *                        char *msg, int msglen);
 * returns 0 on success, non-zero on failure (msg gets a one-line status).
 */
#ifdef _MSC_VER
#include <iostream>
#include <complex>

extern "C" {
#include "snp2va.h"
#include <string.h>
};
#else
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>
#include <complex.h>
#include "snp2va.h"
#endif

#ifdef _MSC_VER
typedef std::complex<double> cplx;
#define cabs abs
#define cpow pow
#define cimag imag
#define creal real
#define cexp exp
#define I cplx(0.0, 1.0)
#define strcasecmp _stricmp
#else
typedef double _Complex cplx;
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ============================ linear algebra ============================ */

/* Least-squares min||A x - b|| for a REAL system via Householder QR. A is
 * row-major m*n, b length m, x length n. Returns 0 on ok. Handles the
 * underdetermined case m<n safely: the Householder sweep triangularizes only the
 * first min(m,n) columns, and the back-substitution below leaves any unknown with
 * no constraining row (i>=m) at zero rather than reading past the m-row A and b
 * (a heap-buffer-overflow reachable from `pre_snp` on a Touchstone file with very
 * few frequency points, where the vector fit's stacked system has fewer rows than
 * poles). For the normal overdetermined path (m>=n) the guard never fires. */
static int lstsq_real(double *A, double *b, int m, int n, double *x)
{
    int i, j, k;
    for (k = 0; k < n; k++) {
        double norm = 0.0;
        for (i = k; i < m; i++) norm += A[i*n+k]*A[i*n+k];
        norm = sqrt(norm);
        if (norm == 0.0) continue;
        double alpha = (A[k*n+k] >= 0.0) ? -norm : norm;
        double *v = (double*) calloc((size_t) m, sizeof(double));
        v[k] = A[k*n+k] - alpha;
        for (i = k+1; i < m; i++) v[i] = A[i*n+k];
        double vn2 = 0.0;
        for (i = k; i < m; i++) vn2 += v[i]*v[i];
        if (vn2 == 0.0) { free(v); continue; }
        for (j = k; j < n; j++) {
            double s = 0.0;
            for (i = k; i < m; i++) s += v[i]*A[i*n+j];
            s = s*2.0/vn2;
            for (i = k; i < m; i++) A[i*n+j] -= s*v[i];
        }
        double s = 0.0;
        for (i = k; i < m; i++) s += v[i]*b[i];
        s = s*2.0/vn2;
        for (i = k; i < m; i++) b[i] -= s*v[i];
        free(v);
    }
    for (i = n-1; i >= 0; i--) {
        if (i >= m) { x[i] = 0.0; continue; }  /* unknown i has no constraining row */
        double acc = b[i];
        for (j = i+1; j < n; j++) acc -= A[i*n+j]*x[j];
        x[i] = (A[i*n+i] != 0.0) ? acc/A[i*n+i] : 0.0;
    }
    return 0;
}

/* In-place inverse of an n x n complex matrix (row-major), Gauss-Jordan w/ pivot.
 * Returns 0 on ok, 1 if singular. */
static int mat_inv_c(cplx *M, int n)
{
    int i, j, c, p;
    cplx *A = (cplx*) malloc((size_t) n*2*n*sizeof(cplx));
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) A[i*2*n+j] = M[i*n+j];
        for (j = 0; j < n; j++) A[i*2*n+n+j] = (i==j) ? 1.0 : 0.0;
    }
    for (c = 0; c < n; c++) {
        p = c; double best = cabs(A[c*2*n+c]);
        for (i = c+1; i < n; i++) { double v = cabs(A[i*2*n+c]); if (v > best) { best = v; p = i; } }
        if (best == 0.0) { free(A); return 1; }
        if (p != c) for (j = 0; j < 2*n; j++) { cplx t = A[c*2*n+j]; A[c*2*n+j]=A[p*2*n+j]; A[p*2*n+j]=t; }
        cplx d = A[c*2*n+c];
        for (j = 0; j < 2*n; j++) A[c*2*n+j] /= d;
        for (i = 0; i < n; i++) if (i != c) {
            cplx f = A[i*2*n+c];
            if (f != 0.0) for (j = 0; j < 2*n; j++) A[i*2*n+j] -= f*A[c*2*n+j];
        }
    }
    for (i = 0; i < n; i++) for (j = 0; j < n; j++) M[i*n+j] = A[i*2*n+n+j];
    free(A);
    return 0;
}

/* Monic polynomial (DESCENDING, len nr+1) from roots. coef must hold nr+1. */
static void poly_from_roots(const cplx *roots, int nr, cplx *coef)
{
    int i, k;
    coef[0] = 1.0;
    for (i = 1; i <= nr; i++) coef[i] = 0.0;
    for (k = 0; k < nr; k++) {
        for (i = k+1; i >= 1; i--) coef[i] = coef[i] - roots[k]*coef[i-1];
    }
}

static cplx poly_eval(const cplx *c, int deg, cplx x)
{
    cplx r = 0.0; int i;
    for (i = 0; i <= deg; i++) r = r*x + c[i];
    return r;
}

/* All roots of a DESCENDING-coeff polynomial (deg = len-1) via Durand-Kerner.
 * roots[] must hold deg. Returns 0 on ok. */
static int poly_roots(const cplx *cin, int deg, cplx *roots)
{
    int i, j, it;
    if (deg <= 0) return 0;
    cplx *c = (cplx*) malloc((size_t)(deg+1)*sizeof(cplx));
    for (i = 0; i <= deg; i++) c[i] = cin[i] / cin[0];   /* monic */
    cplx seed = 0.4 + 0.9*I;
    for (i = 0; i < deg; i++) roots[i] = cpow(seed, (double) i);
    for (it = 0; it < 500; it++) {
        double maxstep = 0.0;
        for (i = 0; i < deg; i++) {
            cplx num = poly_eval(c, deg, roots[i]);
            cplx den = 1.0;
            for (j = 0; j < deg; j++) if (j != i) den *= (roots[i]-roots[j]);
            cplx step = (cabs(den) > 1e-300) ? num/den : 0.0;
            roots[i] -= step;
            if (cabs(step) > maxstep) maxstep = cabs(step);
        }
        if (maxstep < 1e-14) break;
    }
    free(c);
    return 0;
}

/* ============================ Touchstone I/O ============================ */

typedef struct { double *freqs; cplx *S; int nf; int N; double z0; char ptype; } TS;

static void ts_free(TS *t) { free(t->freqs); free(t->S); }

/* returns 0 on ok */
static int parse_touchstone(const char *fn, TS *out, char *msg, int msglen)
{
    FILE *f = fopen(fn, "r");
    if (!f) { snprintf(msg, (size_t) msglen, "cannot open '%s'", fn); return 1; }
    double fmul = 1e9, z0 = 50.0; char ptype = 'S'; char fmt[3] = "MA";
    /* collect all numeric tokens after the '#' options line(s) */
    double *nums = NULL; long ncap = 0, nn = 0;
    char line[4096];
    while (fgets(line, sizeof line, f)) {
        char *h = strchr(line, '!'); if (h) *h = '\0';
        char *p = line;
        while (*p && isspace((unsigned char)*p)) p++;
        if (*p == '\0') continue;
        if (*p == '#') {
            char *tok = strtok(p+1, " \t\r\n");
            while (tok) {
                if (!strcasecmp(tok,"HZ")) fmul=1.0;
                else if (!strcasecmp(tok,"KHZ")) fmul=1e3;
                else if (!strcasecmp(tok,"MHZ")) fmul=1e6;
                else if (!strcasecmp(tok,"GHZ")) fmul=1e9;
                else if (!strcasecmp(tok,"S")||!strcasecmp(tok,"Y")||!strcasecmp(tok,"Z")) ptype=(char)toupper((unsigned char)tok[0]);
                else if (!strcasecmp(tok,"MA")||!strcasecmp(tok,"DB")||!strcasecmp(tok,"RI")) { fmt[0]=(char)toupper((unsigned char)tok[0]); fmt[1]=(char)toupper((unsigned char)tok[1]); }
                else if (!strcasecmp(tok,"R")) { char *z=strtok(NULL," \t\r\n"); if (z) z0=atof(z); }
                tok = strtok(NULL, " \t\r\n");
            }
            continue;
        }
        /* numeric data line */
        char *tok = strtok(p, " \t\r\n");
        while (tok) {
            char *end; double v = strtod(tok, &end);
            if (end != tok) {
                if (nn >= ncap) { ncap = ncap ? ncap*2 : 1024; nums = (double*) realloc(nums, (size_t) ncap*sizeof(double)); }
                nums[nn++] = v;
            }
            tok = strtok(NULL, " \t\r\n");
        }
    }
    fclose(f);
    /* infer port count N: try the file extension .sNp, else brute force */
    int N = 0;
    const char *dot = strrchr(fn, '.');
    if (dot && (dot[1]=='s'||dot[1]=='S') && (fn[strlen(fn)-1]=='p'||fn[strlen(fn)-1]=='P')) {
        N = atoi(dot+2);
        /* Enhancement-227: reject an implausible port count from the filename
         * (e.g. `.s2147483647p`). N is stored in out->N and used to size the
         * downstream N x N vector fit; a huge N over-allocates / overflows and
         * corrupts the heap. Real Touchstone files have few ports -- above the
         * brute-force limit, drop back to inferring N from the data. */
        if (N > 512)
            N = 0;
    }
    if (N <= 0) {
        int c;
        for (c = 1; c <= 512; c++) if (nn % (1 + 2*c*c) == 0) { N = c; break; }
    }
    if (N <= 0) { free(nums); snprintf(msg,(size_t)msglen,"cannot determine port count"); return 1; }
    int rec = 1 + 2*N*N;
    int nf = (int)(nn / rec);
    if (nf < 2) { free(nums); snprintf(msg,(size_t)msglen,"too few frequency points (%d)", nf); return 1; }
    out->freqs = (double*) malloc((size_t) nf*sizeof(double));
    out->S = (cplx*) malloc((size_t) nf*N*N*sizeof(cplx));
    out->nf = nf; out->N = N; out->z0 = z0; out->ptype = ptype;
    cplx *pv = (cplx*) malloc((size_t) N*N*sizeof(cplx));   /* heap: N may be large */
    int r, kk;
    for (r = 0; r < nf; r++) {
        double *chunk = nums + (long) r*rec;
        out->freqs[r] = chunk[0]*fmul;
        double *vals = chunk+1;
        for (kk = 0; kk < N*N; kk++) {
            double a = vals[2*kk], b = vals[2*kk+1];
            if (!strcmp(fmt,"MA")) pv[kk] = a*cexp(I*b*M_PI/180.0);
            else if (!strcmp(fmt,"DB")) pv[kk] = pow(10.0,a/20.0)*cexp(I*b*M_PI/180.0);
            else pv[kk] = a + I*b;
        }
        /* Touchstone: N=2 order is S11 S21 S12 S22; general is row-major */
        cplx *M = out->S + (long) r*N*N;
        if (N == 2) { M[0]=pv[0]; M[2]=pv[1]; M[1]=pv[2]; M[3]=pv[3]; }
        else for (kk = 0; kk < N*N; kk++) M[kk] = pv[kk];
    }
    free(pv); free(nums);
    return 0;
}

/* S/Y/Z -> Y (row-major per frequency), Yout must hold nf*N*N. */
static int to_Y(const TS *t, cplx *Yout)
{
    int N = t->N, r, i, j;
    cplx *tmp = (cplx*) malloc((size_t) N*N*sizeof(cplx));
    for (r = 0; r < t->nf; r++) {
        const cplx *M = t->S + (long) r*N*N;
        cplx *Y = Yout + (long) r*N*N;
        if (t->ptype == 'Y') { for (i=0;i<N*N;i++) Y[i]=M[i]; }
        else if (t->ptype == 'Z') { for (i=0;i<N*N;i++) Y[i]=M[i]; if (mat_inv_c(Y,N)) { free(tmp); return 1; } }
        else { /* S -> Y = (1/z0)(I-S)(I+S)^-1 */
            cplx *IpS = tmp;
            for (i=0;i<N;i++) for (j=0;j<N;j++) IpS[i*N+j] = ((i==j)?1.0:0.0) + M[i*N+j];
            if (mat_inv_c(IpS,N)) { free(tmp); return 1; }
            for (i=0;i<N;i++) for (j=0;j<N;j++) {
                cplx acc = 0.0; int k;
                for (k=0;k<N;k++) acc += (((i==k)?1.0:0.0) - M[i*N+k]) * IpS[k*N+j];
                Y[i*N+j] = acc / t->z0;
            }
        }
    }
    free(tmp);
    return 0;
}

/* ============================ vector fitting ============================ */

/* complex partial-fraction basis (real-valued cc combos), Ns x Np, row-major */
static void build_basis(const cplx *s, int Ns, const cplx *poles, int Np, cplx *A)
{
    int r, i;
    for (r = 0; r < Ns; r++) {
        i = 0;
        while (i < Np) {
            if (fabs(cimag(poles[i])) < 1e-9*fabs(creal(poles[i]))+1e-30) {
                A[r*Np+i] = 1.0/(s[r]-poles[i]); i += 1;
            } else {
                cplx p = poles[i];
                A[r*Np+i]   = 1.0/(s[r]-p) + 1.0/(s[r]-conj(p));
                A[r*Np+i+1] = I/(s[r]-p) - I/(s[r]-conj(p));
                i += 2;
            }
        }
    }
}

/* complex residues from real ctil coeffs, per real/cc layout */
static void ctil_to_cres(const double *ctil, const cplx *poles, int Np, cplx *cres)
{
    int i = 0;
    while (i < Np) {
        if (fabs(cimag(poles[i])) < 1e-9*fabs(creal(poles[i]))+1e-30) { cres[i] = ctil[i]; i += 1; }
        else { cres[i] = ctil[i] + I*ctil[i+1]; cres[i+1] = conj(cres[i]); i += 2; }
    }
}

/* Canonicalize a pole set into the layout every consumer here assumes: real poles
 * (exact Im=0) first, then complex poles as ADJACENT exact conjugate pairs. The
 * Durand-Kerner roots of the (real) sigma numerator are conjugate-symmetric in
 * theory, but numerical noise can leave a "pair" split or a near-real pole with a
 * tiny nonzero imag -- which makes build_basis / ctil_to_cres walk one slot past
 * the array (heap overflow). Rebuilding pairs from the Im>0 representatives (using
 * exact conj) guarantees the structure and cleans the asymmetry. */
static void canon_poles(cplx *p, int Np)
{
    int i, no = 0;
    cplx *out = (cplx*) malloc((size_t) Np*sizeof(cplx));
    for (i = 0; i < Np; i++)
        if (fabs(cimag(p[i])) < 1e-6*cabs(p[i])) p[i] = creal(p[i]);   /* snap near-real */
    for (i = 0; i < Np; i++)
        if (cimag(p[i]) == 0.0 && no < Np) out[no++] = p[i];           /* reals first */
    for (i = 0; i < Np; i++)
        if (cimag(p[i]) > 0.0) {                                       /* one per pair */
            if (no+1 < Np) { out[no++] = p[i]; out[no++] = conj(p[i]); }
            else if (no < Np) out[no++] = creal(p[i]);                 /* no room -> real */
        }
    while (no < Np) { out[no] = -fabs(creal(p[no])) ; no++; }          /* pad (safety) */
    for (i = 0; i < Np; i++) p[i] = out[i];
    free(out);
}

/* Householder-reduce B (m x nB, row-major) in place, applying every reflector to
 * the tail T (m x nT) and rhs g (m). After this the rows [nB..m) of T and g are a
 * reduced least-squares system in the remaining (shared) unknowns only -- the per-
 * element unknowns spanned by B have been projected out. (Fast Vector Fitting:
 * Deschrijver et al. 2008 -- avoids ever forming the full block-arrow matrix.) */
static void hh_reduce(double *B, double *T, double *g, int m, int nB, int nT)
{
    int i, j, k;
    double *v = (double*) malloc((size_t) m * sizeof(double));
    for (k = 0; k < nB; k++) {
        double norm = 0.0;
        for (i = k; i < m; i++) norm += B[i*nB+k]*B[i*nB+k];
        norm = sqrt(norm);
        if (norm == 0.0) continue;
        double alpha = (B[k*nB+k] >= 0.0) ? -norm : norm;
        for (i = 0; i < k; i++) v[i] = 0.0;
        v[k] = B[k*nB+k] - alpha;
        for (i = k+1; i < m; i++) v[i] = B[i*nB+k];
        double vn2 = 0.0;
        for (i = k; i < m; i++) vn2 += v[i]*v[i];
        if (vn2 == 0.0) continue;
        for (j = k; j < nB; j++) {
            double sdot = 0.0; for (i = k; i < m; i++) sdot += v[i]*B[i*nB+j];
            sdot = sdot*2.0/vn2; for (i = k; i < m; i++) B[i*nB+j] -= sdot*v[i];
        }
        for (j = 0; j < nT; j++) {
            double sdot = 0.0; for (i = k; i < m; i++) sdot += v[i]*T[i*nT+j];
            sdot = sdot*2.0/vn2; for (i = k; i < m; i++) T[i*nT+j] -= sdot*v[i];
        }
        { double sdot = 0.0; for (i = k; i < m; i++) sdot += v[i]*g[i];
          sdot = sdot*2.0/vn2; for (i = k; i < m; i++) g[i] -= sdot*v[i]; }
    }
    free(v);
}

/* One vector-fit run (fixed pole count) over the element indices in elems[0..Ne).
 * s,F normalized; F is [N*N][Ns], res/d/e are written only for the listed elements.
 * Common poles are identified with the FAST (block-reduced) pole solve, so cost is
 * O(Ne*Ns*Np^2) with O(Ne*Ns*Np) memory instead of the O(Ne^2) dense stack. Stops
 * early once the poles stop moving. */
static void vector_fit(const cplx *s, int Ns, const cplx *F,
                       const int *elems, int Ne, int Np,
                       cplx *poles, cplx *res, double *d, double *e, int maxiter)
{
    int iter, i, j, k, r, ke;
    cplx *A = (cplx*) malloc((size_t) Ns*Np*sizeof(cplx));
    int m = 2*Ns, nB = Np+2, nT = Np, redrows = 2*Ns - (Np+2);
    if (redrows < 0) redrows = 0;
    long stackrows = (long) redrows * Ne;
    double *SC = (double*) malloc((size_t) stackrows * Np * sizeof(double));
    double *Sb = (double*) malloc((size_t) stackrows * sizeof(double));
    double *B  = (double*) malloc((size_t) m * nB * sizeof(double));
    double *T  = (double*) malloc((size_t) m * nT * sizeof(double));
    double *g  = (double*) malloc((size_t) m * sizeof(double));

    for (iter = 0; iter < maxiter; iter++) {
        build_basis(s, Ns, poles, Np, A);
        long sr = 0;
        for (ke = 0; ke < Ne; ke++) {
            k = elems[ke];
            for (r = 0; r < Ns; r++) {
                cplx Fkr = F[(long)k*Ns + r];
                for (j = 0; j < Np; j++) {
                    cplx a = A[r*Np+j], neg = -Fkr*a;
                    B[r*nB + j]      = creal(a);   B[(r+Ns)*nB + j]   = cimag(a);
                    T[r*nT + j]      = creal(neg); T[(r+Ns)*nT + j]   = cimag(neg);
                }
                B[r*nB + Np]     = 1.0;         B[(r+Ns)*nB + Np]    = 0.0;         /* d */
                B[r*nB + Np+1]   = creal(s[r]); B[(r+Ns)*nB + Np+1]  = cimag(s[r]); /* e*s */
                g[r]             = creal(Fkr);  g[r+Ns]              = cimag(Fkr);
            }
            hh_reduce(B, T, g, m, nB, nT);
            for (i = nB; i < m; i++) {
                for (j = 0; j < Np; j++) SC[sr*Np + j] = T[i*nT + j];
                Sb[sr] = g[i];
                sr++;
            }
        }
        double *ctil = (double*) calloc((size_t) Np, sizeof(double));
        lstsq_real(SC, Sb, (int) sr, Np, ctil);
        cplx *cres = (cplx*) malloc((size_t) Np*sizeof(cplx));
        ctil_to_cres(ctil, poles, Np, cres);
        /* relocate: roots of  D(s) + sum cres_i * D(s)/(s-a_i) */
        cplx *D = (cplx*) malloc((size_t)(Np+1)*sizeof(cplx));
        poly_from_roots(poles, Np, D);
        cplx *numsig = (cplx*) malloc((size_t)(Np+1)*sizeof(cplx));
        for (i = 0; i <= Np; i++) numsig[i] = D[i];
        cplx *Di = (cplx*) malloc((size_t) Np*sizeof(cplx));
        cplx *sub = (cplx*) malloc((size_t) Np*sizeof(cplx));   /* poles minus i */
        for (i = 0; i < Np; i++) {
            int t = 0; for (j = 0; j < Np; j++) if (j != i) sub[t++] = poles[j];
            poly_from_roots(sub, Np-1, Di);                     /* len Np, degree Np-1 */
            for (j = 0; j < Np; j++) numsig[j+1] += cres[i]*Di[j];
        }
        cplx *newp = (cplx*) malloc((size_t) Np*sizeof(cplx));
        poly_roots(numsig, Np, newp);
        for (i = 0; i < Np; i++) if (creal(newp[i]) > 0) newp[i] = -creal(newp[i]) + I*cimag(newp[i]);
        /* sort: real poles first, then by real, imag (keeps cc pairs adjacent-ish) */
        for (i = 0; i < Np; i++) for (j = i+1; j < Np; j++) {
            int swap = 0;
            double ki = (fabs(cimag(newp[i]))>1e-6)?1:0, kj = (fabs(cimag(newp[j]))>1e-6)?1:0;
            if (kj < ki) swap = 1;
            else if (kj == ki) { if (creal(newp[j]) < creal(newp[i]) - 1e-30) swap = 1;
                                 else if (fabs(creal(newp[j])-creal(newp[i]))<1e-30 && cimag(newp[j])<cimag(newp[i])) swap = 1; }
            if (swap) { cplx t = newp[i]; newp[i]=newp[j]; newp[j]=t; }
        }
        canon_poles(newp, Np);   /* exact adjacent conjugate pairs (prevents OOB) */
        double mv = 0.0;   /* max relative pole movement -> convergence */
        for (i = 0; i < Np; i++) {
            double dm = cabs(newp[i]-poles[i])/(cabs(poles[i])+1e-300);
            if (dm > mv) mv = dm;
        }
        for (i = 0; i < Np; i++) poles[i] = newp[i];
        free(ctil); free(cres); free(D); free(numsig); free(Di); free(sub); free(newp);
        if (mv < 1e-4) break;              /* poles settled -- stop early (Fix #3) */
    }
    /* final residues (fixed poles), per fitted element */
    build_basis(s, Ns, poles, Np, A);
    for (ke = 0; ke < Ne; ke++) {
        k = elems[ke];
        int ncol = Np+2, nrow = Ns;
        double *M = (double*) calloc((size_t)(2*nrow)*ncol, sizeof(double));
        double *bb = (double*) calloc((size_t)(2*nrow), sizeof(double));
        for (r = 0; r < Ns; r++) {
            cplx Fkr = F[(long)k*Ns+r];
            for (j = 0; j < Np; j++) { M[r*ncol+j]=creal(A[r*Np+j]); M[(r+nrow)*ncol+j]=cimag(A[r*Np+j]); }
            M[r*ncol+Np]=1.0;
            M[r*ncol+Np+1]=creal(s[r]); M[(r+nrow)*ncol+Np+1]=cimag(s[r]);
            bb[r]=creal(Fkr); bb[r+nrow]=cimag(Fkr);
        }
        double *x = (double*) calloc((size_t) ncol, sizeof(double));
        lstsq_real(M, bb, 2*nrow, ncol, x);
        cplx *cr = (cplx*) malloc((size_t) Np*sizeof(cplx));
        ctil_to_cres(x, poles, Np, cr);
        for (j = 0; j < Np; j++) res[(long)k*Np+j] = cr[j];
        d[k] = x[Np]; e[k] = x[Np+1];
        free(M); free(bb); free(x); free(cr);
    }
    free(A); free(SC); free(Sb); free(B); free(T); free(g);
}

/* Reciprocal network? (S/Y symmetric -> fit only the upper triangle.) F is the
 * Y data [N*N][Ns]; compare Y_ij vs Y_ji across a frequency subset. */
static int is_reciprocal(const cplx *F, int N, int Ns)
{
    double maxd = 0.0, maxv = 0.0;
    int i, j, r;
    int step = Ns/16 > 0 ? Ns/16 : 1;
    for (i = 0; i < N; i++) for (j = i+1; j < N; j++)
        for (r = 0; r < Ns; r += step) {
            cplx a = F[(long)(i*N+j)*Ns+r], b = F[(long)(j*N+i)*Ns+r];
            double dd = cabs(a-b), va = cabs(a);
            if (dd > maxd) maxd = dd;
            if (va > maxv) maxv = va;
        }
    return maxd <= 1e-6*(maxv+1e-300);
}

static void seed_poles(double fmin, double fmax, int npair, cplx *p)
{
    int k;
    for (k = 0; k < npair; k++) {
        double beta = (npair==1) ? 2*M_PI*sqrt(fmin*fmax)
                                 : 2*M_PI*fmin*pow(fmax/fmin, (double)k/(npair-1));
        double a = beta/100.0;
        p[2*k]   = -a + I*beta;
        p[2*k+1] = -a - I*beta;
    }
}

/* ============================ emit VA ============================ */
static void emit_arr(FILE *f, const double *v, int n)
{
    int i; fprintf(f, "'{");
    for (i = 0; i < n; i++) {
        char buf[64];
        snprintf(buf, sizeof buf, "%.12g", v[i]);
        /* Force a REAL literal: `%.12g` prints 1.0 as "1", and OpenVAF crashes on an
         * integer literal inside a laplace_nd coefficient array assigned to a
         * variable (index-out-of-bounds in its lowering). Append ".0" if needed. */
        fprintf(f, "%s%s%s", i?", ":"", buf, strpbrk(buf, ".eE") ? "" : ".0");
    }
    fprintf(f, "}");
}

/* term separator inside one `I(p) <+ ...;` contribution */
static void term_sep(FILE *f, int *first)
{
    if (*first) *first = 0;
    else fprintf(f, "\n                 + ");
}

/* Symmetric eigen-decomposition via cyclic Jacobi. A (n x n, row-major) is
 * overwritten; eigenvalues -> w, orthonormal eigenvectors (columns) -> V. */
static void jacobi_sym(double *A, int n, double *w, double *V)
{
    int i, j, p, q, sweep;
    for (i = 0; i < n; i++) { for (j = 0; j < n; j++) V[i*n+j] = (i==j)?1.0:0.0; }
    for (sweep = 0; sweep < 100; sweep++) {
        double off = 0.0;
        for (p = 0; p < n; p++) for (q = p+1; q < n; q++) off += A[p*n+q]*A[p*n+q];
        if (off < 1e-300) break;
        for (p = 0; p < n; p++) for (q = p+1; q < n; q++) {
            double apq = A[p*n+q];
            if (fabs(apq) < 1e-300) continue;
            double app = A[p*n+p], aqq = A[q*n+q];
            double phi = 0.5*(aqq-app)/apq;
            double t = (phi>=0?1.0:-1.0)/(fabs(phi)+sqrt(phi*phi+1.0));
            double c = 1.0/sqrt(t*t+1.0), sn = t*c;
            for (i = 0; i < n; i++) {
                double aip = A[i*n+p], aiq = A[i*n+q];
                A[i*n+p] = c*aip - sn*aiq; A[i*n+q] = sn*aip + c*aiq;
            }
            for (i = 0; i < n; i++) {
                double api = A[p*n+i], aqi = A[q*n+i];
                A[p*n+i] = c*api - sn*aqi; A[q*n+i] = sn*api + c*aqi;
            }
            for (i = 0; i < n; i++) {
                double vip = V[i*n+p], viq = V[i*n+q];
                V[i*n+p] = c*vip - sn*viq; V[i*n+q] = sn*vip + c*viq;
            }
        }
    }
    for (i = 0; i < n; i++) w[i] = A[i*n+i];
}

/* Project the improper (e*s) capacitance matrix onto the symmetric PSD cone.
 * Each Y_ij is fit independently, so E=[e_ij] carries no passivity constraint;
 * a negative eigenvalue is a "negative capacitance" that makes the transient DAE
 * unstable (it diverges) even though every pole is in the LHP and AC is exact.
 * Symmetrizing and clamping negative eigenvalues to 0 yields a passive C-matrix
 * -> stable transient, while genuinely-improper (shunt-C) networks keep their
 * positive eigenvalues. e[] is indexed i*N+j. */
static void psd_project_E(double *e, int N)
{
    double *A = (double*) malloc((size_t) N*N*sizeof(double));
    double *V = (double*) malloc((size_t) N*N*sizeof(double));
    double *w = (double*) malloc((size_t) N*sizeof(double));
    int i, j, k;
    for (i = 0; i < N; i++) for (j = 0; j < N; j++) A[i*N+j] = 0.5*(e[i*N+j]+e[j*N+i]);
    jacobi_sym(A, N, w, V);
    for (i = 0; i < N; i++) for (j = 0; j < N; j++) {
        double acc = 0.0;
        for (k = 0; k < N; k++) { double wk = w[k] > 0.0 ? w[k] : 0.0; acc += V[i*N+k]*wk*V[j*N+k]; }
        e[i*N+j] = acc;
    }
    free(A); free(V); free(w);
}

/* One-sided Jacobi SVD of a square real n x n matrix: A = U * diag(S) * V^T,
 * S descending. U, S, V preallocated (n*n, n, n*n). Robust, no external deps;
 * used to detect and factor low-rank coupling in the emit (Enhancement-205). */
static void jacobi_svd(const double *Ain, int n, double *U, double *S, double *V)
{
    int i, p, q, sweep;
    for (i = 0; i < n*n; i++) U[i] = Ain[i];
    for (i = 0; i < n; i++) { int t; for (t = 0; t < n; t++) V[i*n+t] = (i==t)?1.0:0.0; }
    for (sweep = 0; sweep < 60; sweep++) {
        int rotated = 0;
        for (p = 0; p < n-1; p++) for (q = p+1; q < n; q++) {
            double alpha=0, beta=0, gamma=0;
            for (i = 0; i < n; i++) { double ap=U[i*n+p], aq=U[i*n+q];
                alpha += ap*ap; beta += aq*aq; gamma += ap*aq; }
            if (fabs(gamma) <= 1e-15*sqrt(alpha*beta)) continue;
            rotated = 1;
            double zeta = (beta - alpha) / (2.0*gamma);
            double t = (zeta>=0?1.0:-1.0) / (fabs(zeta) + sqrt(1.0+zeta*zeta));
            double c = 1.0/sqrt(1.0+t*t), sn = c*t;
            for (i = 0; i < n; i++) {
                double ap=U[i*n+p], aq=U[i*n+q];
                U[i*n+p] = c*ap - sn*aq; U[i*n+q] = sn*ap + c*aq;
                double vp=V[i*n+p], vq=V[i*n+q];
                V[i*n+p] = c*vp - sn*vq; V[i*n+q] = sn*vp + c*vq;
            }
        }
        if (!rotated) break;
    }
    for (q = 0; q < n; q++) {
        double nrm = 0.0; for (i = 0; i < n; i++) nrm += U[i*n+q]*U[i*n+q];
        nrm = sqrt(nrm); S[q] = nrm;
        if (nrm > 1e-300) for (i = 0; i < n; i++) U[i*n+q] /= nrm;
    }
    for (p = 0; p < n-1; p++) {   /* sort columns by S descending (n small) */
        int mx = p; for (q = p+1; q < n; q++) if (S[q] > S[mx]) mx = q;
        if (mx != p) { double t=S[p]; S[p]=S[mx]; S[mx]=t;
            for (i = 0; i < n; i++) { double a=U[i*n+p]; U[i*n+p]=U[i*n+mx]; U[i*n+mx]=a;
                                      double b=V[i*n+p]; V[i*n+p]=V[i*n+mx]; V[i*n+mx]=b; } }
    }
}

/* Emit the "num, den" coefficient arrays of one pole section's laplace_nd:
 * kind 0 = real pole (1/(s-p)); 1 = conj-pair "cos" (s-sigma)/D; 2 = "sin" om/D. */
static void emit_filter(FILE *fo, cplx pole, int kind)
{
    if (kind == 0) {
        double num[1] = { 1.0 }, de[2] = { -creal(pole), 1.0 };
        emit_arr(fo, num, 1); fprintf(fo, ", "); emit_arr(fo, de, 2);
    } else {
        double sig = creal(pole), om = cimag(pole);
        double de[3] = { sig*sig+om*om, -2.0*sig, 1.0 };
        if (kind == 1) { double num[2] = { -sig, 1.0 }; emit_arr(fo, num, 2); }
        else           { double num[1] = { om };        emit_arr(fo, num, 1); }
        fprintf(fo, ", "); emit_arr(fo, de, 3);
    }
}

/* ============================ public API ============================ */
/* Shared front half: parse Touchstone, S->Y, common-pole vector fit with order
 * selection, reciprocal mirror, PSD-project E. On success returns 0 and hands the
 * caller freshly-owned fit arrays P[Np], res[N*N*Np], d[N*N], e[N*N] (caller frees
 * them); the parse scaffolding is freed here. Layout is exactly what both emitters
 * and the native `.nport` device expect: poles canonicalized (real first, then
 * adjacent conjugate pairs); res indexed (i*N+j)*Np+k. Leak-tolerant for discarded
 * candidate fits, as before -- a one-shot conversion tool. */
static int snp_fit(const char *snpfile, int *pN, int *pNp,
                   cplx **pP, cplx **pRes, double **pD, double **pE,
                   double *pErr, char *msg, int msglen)
{
    TS ts; int i, j, k, r;
    if (parse_touchstone(snpfile, &ts, msg, msglen)) return 1;
    int N = ts.N, nf = ts.nf;
    cplx *Y = (cplx*) malloc((size_t) nf*N*N*sizeof(cplx));
    if (to_Y(&ts, Y)) { snprintf(msg,(size_t)msglen,"S->Y conversion failed (singular)"); ts_free(&ts); free(Y); return 1; }
    /* Y entries in row-major i*N+j order, each a length-nf vector */
    int Nf = N*N;
    cplx *s = (cplx*) malloc((size_t) nf*sizeof(cplx));
    for (r = 0; r < nf; r++) s[r] = 2*M_PI*ts.freqs[r]*I;
    double wn = sqrt(cabs(s[0])*cabs(s[nf-1]));
    cplx *sn = (cplx*) malloc((size_t) nf*sizeof(cplx));
    for (r = 0; r < nf; r++) sn[r] = s[r]/wn;
    cplx *F = (cplx*) malloc((size_t) Nf*nf*sizeof(cplx));
    for (i = 0; i < N; i++) for (j = 0; j < N; j++) for (r = 0; r < nf; r++)
        F[(long)(i*N+j)*nf+r] = Y[(long)r*N*N + i*N+j];

    /* reciprocity: a passive network gives symmetric Y, so fit only the upper
     * triangle (N(N+1)/2 elements) and mirror -- ~2x fewer LS solves (Fix #2). */
    int reciprocal = is_reciprocal(F, N, nf);
    int *elems = (int*) malloc((size_t) Nf*sizeof(int));
    int Ne = 0;
    if (reciprocal) { for (i = 0; i < N; i++) for (j = i; j < N; j++) elems[Ne++] = i*N+j; }
    else            { for (i = 0; i < Nf; i++) elems[Ne++] = i; }

    /* ---- order selection: climb, keep best STABLE fit, knee near a floor ---- */
    double fmin = ts.freqs[0], fmax = ts.freqs[nf-1], tol = 1e-3;
    cplx *bestP=NULL,*bestRes=NULL; double *bestD=NULL,*bestE=NULL; int bestNp=0; double bestErr=1e300;
    cplx *prevP=NULL,*prevRes=NULL; double *prevD=NULL,*prevE=NULL; int prevNp=0; double prevErr=-1, firstErr=-1;
    int chosenP=0; cplx *chP=NULL,*chRes=NULL; double *chD=NULL,*chE=NULL;
    int npair;
    for (npair = 1; npair <= 12; npair++) {
        int Np = 2*npair;
        cplx *P = (cplx*) malloc((size_t) Np*sizeof(cplx));
        cplx *res = (cplx*) malloc((size_t) Nf*Np*sizeof(cplx));
        double *dd = (double*) malloc((size_t) Nf*sizeof(double));
        double *ee = (double*) malloc((size_t) Nf*sizeof(double));
        cplx *p0 = (cplx*) malloc((size_t) Np*sizeof(cplx));
        seed_poles(fmin, fmax, npair, p0);
        for (i = 0; i < Np; i++) P[i] = p0[i]/wn;
        vector_fit(sn, nf, F, elems, Ne, Np, P, res, dd, ee, 10);
        /* un-normalize (fitted elements only) */
        for (i = 0; i < Np; i++) P[i] *= wn;
        for (int ei = 0; ei < Ne; ei++) { k = elems[ei]; for (i = 0; i < Np; i++) res[(long)k*Np+i] *= wn; ee[k] /= wn; }
        /* rms rel error + stability (over fitted elements; Y symmetric so representative) */
        double err = 0.0; int stable = 1;
        for (i = 0; i < Np; i++) if (creal(P[i]) > 1e-6) stable = 0;
        for (int ei = 0; ei < Ne; ei++) {
            k = elems[ei];
            double numr=0, denr=0;
            for (r = 0; r < nf; r++) {
                cplx fit = dd[k] + s[r]*ee[k];
                for (i = 0; i < Np; i++) fit += res[(long)k*Np+i]/(s[r]-P[i]);
                cplx dif = fit - F[(long)k*nf+r];
                numr += creal(dif)*creal(dif)+cimag(dif)*cimag(dif);
                cplx fv = F[(long)k*nf+r];
                denr += creal(fv)*creal(fv)+cimag(fv)*cimag(fv);
            }
            double e2 = sqrt(numr)/(sqrt(denr)+1e-300);
            if (e2 > err) err = e2;
        }
        if (!(err==err)) stable = 0;   /* NaN */
        free(p0);
                if (firstErr < 0) firstErr = err;
        int keep_best = stable && err < bestErr;
        if (keep_best) {
            /* Don't free the old best buffers if `prev` still aliases them (that
             * happens after any prior keep_best iteration, where best==prev==P):
             * the prev-shift below frees them exactly once. Freeing here would
             * leave prev dangling and double-free at line ~490. */
            if (bestP != prevP) { free(bestP);free(bestRes);free(bestD);free(bestE); }
            bestP=P;bestRes=res;bestD=dd;bestE=ee;bestNp=Np;bestErr=err;
        }
        if (!stable) { if(!keep_best){free(P);free(res);free(dd);free(ee);} break; }
        if (err < tol) { chosenP=Np; chP=P;chRes=res;chD=dd;chE=ee; if(keep_best){/*owned by best too*/} break; }
        int near_floor = (err < 0.1*firstErr) || (err < 0.05);
        if (prevErr >= 0 && err > 0.7*prevErr && near_floor) {  /* knee at floor -> use prev */
            chosenP=prevNp; chP=prevP;chRes=prevRes;chD=prevD;chE=prevE;
            if (!keep_best) { free(P);free(res);free(dd);free(ee); }
            break;
        }
        /* shift prev <- current (free old prev unless it is the best) */
        if (prevP && prevP!=bestP) { free(prevP);free(prevRes);free(prevD);free(prevE); }
        prevP=P;prevRes=res;prevD=dd;prevE=ee;prevNp=Np;prevErr=err;
    }
    /* pick chosen, else best, else prev */
    cplx *P; cplx *res; double *dd,*ee; int Np;
    if (chP) { P=chP;res=chRes;dd=chD;ee=chE;Np=chosenP; }
    else if (bestP) { P=bestP;res=bestRes;dd=bestD;ee=bestE;Np=bestNp; }
    else { P=prevP;res=prevRes;dd=prevD;ee=prevE;Np=prevNp; }

    /* mirror the fitted upper triangle into the lower one (reciprocal case, Fix #2) */
    if (reciprocal) {
        for (i = 0; i < N; i++) for (j = i+1; j < N; j++) {
            long u = i*N+j, l = j*N+i;
            for (k = 0; k < Np; k++) res[l*Np+k] = res[u*Np+k];
            dd[l] = dd[u]; ee[l] = ee[u];
        }
    }

    /* force the improper (e*s) capacitance matrix passive so transient is stable */
    psd_project_E(ee, N);

    *pN = N; *pNp = Np; *pP = P; *pRes = res; *pD = dd; *pE = ee; *pErr = bestErr;
    free(Y); free(s); free(sn); free(F); free(elems); ts_free(&ts);
    return 0;
}

/* ---------- Touchstone -> Verilog-A (structured laplace_nd realization) ---------- */
int snp2va_convert(const char *snpfile, const char *vafile, const char *module,
                   char *msg, int msglen)
{
    int N, Np, i, j, k;
    cplx *P, *res; double *dd, *ee, bestErr;
    if (snp_fit(snpfile, &N, &Np, &P, &res, &dd, &ee, &bestErr, msg, msglen)) return 1;

    /* ---- emit VA (shared-pole realization; Fix #4) ----
     * All N^2 elements share the SAME poles, so realize the pole-filters ONCE per
     * input port and form each output current as a cheap weighted sum, instead of
     * one independent laplace_nd bank per element. That is N*Np filter sections and
     * O(N*Np) OSDI state, not O(N^2*Np) -- the difference between a model that
     * compiles/simulates at large N and one that does not. Each section is still a
     * well-conditioned 1st/2nd-order laplace_nd (a real pole -> res/(s-p); a conj
     * pair -> a real "cos" basis (s-sigma)/D and "sin" basis omega/D, with the
     * residue entering as a real weight), so coefficients stay <= O(|p|^2) and the
     * transient is stable. d is a plain conductance, e*s a ddt (PSD-projected). */
    int nsec = 0;
    int *sc_pole = (int*) malloc((size_t) Np*sizeof(int));
    int *sc_kind = (int*) malloc((size_t) Np*sizeof(int));   /* 0 real, 1 cos, 2 sin */
    for (k = 0; k < Np; ) {
        int is_pair = (k+1 < Np) && (fabs(cimag(P[k])) > 1e-6*cabs(P[k]));
        if (!is_pair) { sc_pole[nsec]=k; sc_kind[nsec]=0; nsec++; k += 1; }
        else { sc_pole[nsec]=k; sc_kind[nsec]=1; nsec++;
               sc_pole[nsec]=k; sc_kind[nsec]=2; nsec++; k += 2; }
    }
    /* Enhancement-205: structured (diagonal + low-rank) emit. Build each
     * "channel" weight matrix W (N x N, real) -- the conductance d, the
     * capacitance e, and every pole section's residue weights -- and pick, per
     * channel, the cheaper of a DENSE emit (a term per significant entry; one
     * laplace_nd per input port) or a LOW-RANK emit W = U*V^T (rank r). Because
     * laplace is linear, the low-rank form filters the r COMBINED inputs
     * u_m = sum_j V[j][m]*V(p_j) ONCE (r filters, not N) and distributes them to
     * the outputs via U -- O(N*r) terms and O(r*Np) filters. A full-rank device
     * keeps the dense form (identical model, modulo dropping <1e-8 fit noise);
     * a device whose ports couple through a few shared modes (multi-port
     * filters/cavities/packages) collapses to a compact, fast-compiling model. */
    int force_dense = (getenv("PRE_SNP_DENSE") != NULL);  /* escape hatch / A-B test */
    const double tol_drop = 1e-9;   /* dense: drop only sub-1e-9 fit noise */
    const double tol_rank = 1e-7;   /* low-rank: keep modes above 1e-7 (well below any
                                     * verify tolerance, so a genuinely low-rank device
                                     * with a clean singular-value gap compresses fully
                                     * and is reproduced essentially exactly, while a
                                     * slowly-decaying channel just stays dense). */
    int nchan = 2 + nsec, c, m;
    double **chW  = (double**) malloc((size_t) nchan*sizeof(double*));
    double **chU  = (double**) calloc((size_t) nchan, sizeof(double*));
    double **chV  = (double**) calloc((size_t) nchan, sizeof(double*));
    double  *chMx = (double*)  malloc((size_t) nchan*sizeof(double));
    int *ch_lr   = (int*) calloc((size_t) nchan, sizeof(int));
    int *ch_r    = (int*) calloc((size_t) nchan, sizeof(int));
    int *ch_kind = (int*) malloc((size_t) nchan*sizeof(int)); /* 0=V, 1=ddt, 2=filtered */
    int *ch_sec  = (int*) malloc((size_t) nchan*sizeof(int));
    double *svU=(double*)malloc((size_t)N*N*sizeof(double));
    double *svS=(double*)malloc((size_t)N*sizeof(double));
    double *svV=(double*)malloc((size_t)N*N*sizeof(double));
    int nfilt = 0;
    for (c = 0; c < nchan; c++) {
        double *W = (double*) malloc((size_t) N*N*sizeof(double));
        if (c == 0)      { ch_kind[c]=0; ch_sec[c]=-1; for (i=0;i<N*N;i++) W[i]=dd[i]; }
        else if (c == 1) { ch_kind[c]=1; ch_sec[c]=-1; for (i=0;i<N*N;i++) W[i]=ee[i]; }
        else { int sc=c-2, kk=sc_pole[sc]; ch_kind[c]=2; ch_sec[c]=sc;
            for (i=0;i<N;i++) for (j=0;j<N;j++) { cplx rr=res[(long)(i*N+j)*Np+kk];
                W[i*N+j] = (sc_kind[sc]==0)? creal(rr) : (sc_kind[sc]==1)? 2.0*creal(rr) : -2.0*cimag(rr); } }
        chW[c]=W;
        double mx=0; for (i=0;i<N*N;i++) if (fabs(W[i])>mx) mx=fabs(W[i]);
        chMx[c]=mx;
        if (mx==0.0) continue;                              /* zero channel */
        int nnz=0; for (i=0;i<N*N;i++) if (fabs(W[i])>tol_drop*mx) nnz++;
        jacobi_svd(W, N, svU, svS, svV);
        int rnk=0; for (i=0;i<N;i++) if (svS[i] > tol_rank*svS[0]) rnk++; if (rnk<1) rnk=1;
        int filt = (ch_kind[c]==2);
        long lr_cost = 2L*N*rnk + (filt? rnk : 0);
        long de_cost = (long) nnz + (filt? N : 0);
        if (!force_dense && lr_cost < de_cost) {
            ch_lr[c]=1; ch_r[c]=rnk; nfilt += filt? rnk : 0;
            double *Uc=(double*)malloc((size_t)N*rnk*sizeof(double));
            double *Vc=(double*)malloc((size_t)N*rnk*sizeof(double));
            for (i=0;i<N;i++) for (m=0;m<rnk;m++){ double sq=sqrt(svS[m]);
                Uc[i*rnk+m]=svU[i*N+m]*sq; Vc[i*rnk+m]=svV[i*N+m]*sq; }
            chU[c]=Uc; chV[c]=Vc;
        } else if (filt) nfilt += N;
    }
    free(svU); free(svS); free(svV);

    FILE *fo = fopen(vafile, "w");
    if (!fo) { snprintf(msg,(size_t)msglen,"cannot write '%s'", vafile); return 1; }
    fprintf(fo, "`include \"disciplines.vams\"\n\n");
    fprintf(fo, "// Generated by pre_snp from %s\n", snpfile);
    fprintf(fo, "// %d-port, %d common poles; structured realization (%d laplace_nd filters, AC + transient).\n",
            N, Np, nfilt);
    /* E-243: an explicit `ref` reference terminal (last port), so every branch is
     * V(pi,ref)/I(pi,ref) and the instance line -- `N1 p1..pN ref model` -- is
     * IDENTICAL to the native `nport` device (tie ref to 0 for a ground-referenced
     * Touchstone block). */
    fprintf(fo, "module %s(", module);
    for (i = 0; i < N; i++) fprintf(fo, "%sp%d", i?", ":"", i+1);
    fprintf(fo, ", ref);\n    inout ");
    for (i = 0; i < N; i++) fprintf(fo, "%sp%d", i?", ":"", i+1);
    fprintf(fo, ", ref;\n    electrical ");
    for (i = 0; i < N; i++) fprintf(fo, "%sp%d", i?", ":"", i+1);
    fprintf(fo, ", ref;\n");
    /* variable declarations */
    for (c = 0; c < nchan; c++) {
        if (chMx[c]==0.0) continue;
        if (ch_kind[c]==2 && !ch_lr[c]) {           /* dense filtered: one filter per input */
            fprintf(fo, "    real "); for (j=0;j<N;j++) fprintf(fo, "%sw%d_%d", j?", ":"", c, j); fprintf(fo, ";\n");
        } else if (ch_kind[c]==2 && ch_lr[c]) {     /* low-rank filtered: r combined inputs + r filtered */
            fprintf(fo, "    real "); for (m=0;m<ch_r[c];m++) fprintf(fo, "%su%d_%d, h%d_%d", m?", ":"", c, m, c, m); fprintf(fo, ";\n");
        } else if (ch_kind[c]!=2 && ch_lr[c]) {     /* low-rank algebraic (d/e): r intermediates */
            fprintf(fo, "    real "); for (m=0;m<ch_r[c];m++) fprintf(fo, "%sg%d_%d", m?", ":"", c, m); fprintf(fo, ";\n");
        }
    }
    fprintf(fo, "    analog begin\n");
    /* filter / intermediate assignments */
    for (c = 0; c < nchan; c++) {
        if (chMx[c]==0.0) continue;
        if (ch_kind[c]!=2) {                        /* algebraic: only low-rank needs an intermediate */
            if (!ch_lr[c]) continue;
            for (m=0;m<ch_r[c];m++) {
                int first=1; fprintf(fo, "        g%d_%d = ", c, m);
                for (j=0;j<N;j++){ double v=chV[c][j*ch_r[c]+m];
                    if (fabs(v)>1e-30){ term_sep(fo,&first);
                        if (ch_kind[c]==0) fprintf(fo,"(%.12g)*V(p%d,ref)",v,j+1); else fprintf(fo,"(%.12g)*ddt(V(p%d,ref))",v,j+1); } }
                if (first) fprintf(fo,"0.0"); fprintf(fo,";\n");
            }
            continue;
        }
        int sc=ch_sec[c], kk=sc_pole[sc];
        if (!ch_lr[c]) {                            /* dense: one laplace_nd per input port */
            for (j=0;j<N;j++){ fprintf(fo,"        w%d_%d = laplace_nd(V(p%d,ref), ", c, j, j+1);
                emit_filter(fo, P[kk], sc_kind[sc]); fprintf(fo,");\n"); }
        } else {                                    /* low-rank: r combined inputs, each filtered once */
            for (m=0;m<ch_r[c];m++){
                int first=1; fprintf(fo,"        u%d_%d = ", c, m);
                for (j=0;j<N;j++){ double v=chV[c][j*ch_r[c]+m];
                    if (fabs(v)>1e-30){ term_sep(fo,&first); fprintf(fo,"(%.12g)*V(p%d,ref)",v,j+1); } }
                if (first) fprintf(fo,"0.0"); fprintf(fo,";\n");
                fprintf(fo,"        h%d_%d = laplace_nd(u%d_%d, ", c, m, c, m);
                emit_filter(fo, P[kk], sc_kind[sc]); fprintf(fo,");\n");
            }
        }
    }
    /* output currents */
    for (i = 0; i < N; i++) {
        int first = 1; fprintf(fo, "        I(p%d,ref) <+ ", i+1);
        for (c = 0; c < nchan; c++) {
            if (chMx[c]==0.0) continue;
            if (ch_lr[c]) {
                for (m=0;m<ch_r[c];m++){ double u=chU[c][i*ch_r[c]+m];
                    if (fabs(u)>1e-30){ term_sep(fo,&first);
                        if (ch_kind[c]==2) fprintf(fo,"(%.12g)*h%d_%d",u,c,m); else fprintf(fo,"(%.12g)*g%d_%d",u,c,m); } }
            } else {
                double thr = tol_drop*chMx[c];
                for (j=0;j<N;j++){ double w=chW[c][i*N+j];
                    if (fabs(w)>thr){ term_sep(fo,&first);
                        if (ch_kind[c]==0)      fprintf(fo,"(%.12g)*V(p%d,ref)",w,j+1);
                        else if (ch_kind[c]==1) fprintf(fo,"(%.12g)*ddt(V(p%d,ref))",w,j+1);
                        else                    fprintf(fo,"(%.12g)*w%d_%d",w,c,j); } }
            }
        }
        if (first) fprintf(fo, "0.0"); fprintf(fo, ";\n");
    }
    fprintf(fo, "    end\nendmodule\n");
    fclose(fo);
    for (c=0;c<nchan;c++){ free(chW[c]); free(chU[c]); free(chV[c]); }
    free(chW); free(chU); free(chV); free(chMx); free(ch_lr); free(ch_r); free(ch_kind); free(ch_sec);
    free(sc_pole); free(sc_kind);
    snprintf(msg,(size_t)msglen,"%d-port, %d poles, rms rel err %.2e", N, Np, bestErr<1e300?bestErr:0.0);
    return 0;
}

/* ---------- Touchstone -> native `.nport` fit file (Enhancement-242) ----------
 * Same shared parse+fit, written directly as the compact pole/residue description
 * the built-in n-port device reads -- no Verilog-A, no openvaf-r compile. The
 * device evaluates the identical model Y_ij(s)=d+s*e+sum_k res/(s-p_k), so a
 * `-native` run reproduces a `-osdi` run to fit accuracy. */
int snp2nport_convert(const char *snpfile, const char *nportfile,
                      char *msg, int msglen)
{
    int N, Np, i, j, k;
    cplx *P, *res; double *dd, *ee, bestErr;
    if (snp_fit(snpfile, &N, &Np, &P, &res, &dd, &ee, &bestErr, msg, msglen)) return 1;

    FILE *fo = fopen(nportfile, "w");
    if (!fo) { snprintf(msg,(size_t)msglen,"cannot write '%s'", nportfile); return 1; }
    fprintf(fo, "NPORT 1\n");
    fprintf(fo, "# generated by pre_snp -native from %s\n", snpfile);
    fprintf(fo, "nports %d\nnpoles %d\n", N, Np);
    fprintf(fo, "poles\n");
    for (k = 0; k < Np; k++)
        fprintf(fo, "  %.15e %.15e\n", creal(P[k]), cimag(P[k]));
    fprintf(fo, "d\n");
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) fprintf(fo, " %.15e", dd[i*N+j]);
        fprintf(fo, "\n");
    }
    fprintf(fo, "e\n");
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) fprintf(fo, " %.15e", ee[i*N+j]);
        fprintf(fo, "\n");
    }
    fprintf(fo, "res\n");
    for (i = 0; i < N; i++) for (j = 0; j < N; j++)
        for (k = 0; k < Np; k++) {
            long idx = (long)(i*N+j)*Np + k;
            fprintf(fo, "  %.15e %.15e\n", creal(res[idx]), cimag(res[idx]));
        }
    fclose(fo);
    snprintf(msg,(size_t)msglen,"%d-port, %d poles, rms rel err %.2e", N, Np, bestErr<1e300?bestErr:0.0);
    return 0;
}

#ifdef SNP2VA_TEST
int main(int argc, char **argv)
{
    setbuf(stderr, NULL);
    if (argc < 3) { fprintf(stderr, "usage: %s file.sNp out.va [module]\n", argv[0]); return 2; }
    char msg[256];
    int rc = snp2va_convert(argv[1], argv[2], argc>3?argv[3]:"nport", msg, sizeof msg);
    fprintf(stderr, "snp2va: %s\n", msg);
    return rc;
}
#endif
