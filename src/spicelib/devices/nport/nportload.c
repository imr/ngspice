/**********
Enhancement-242: native n-port device -- DC + transient load, shared helper.

    Y_ij(s) = d_ij + s*e_ij + sum_k res_ijk / (s - p_k)      (all complex)

Port current leaving node i is  I_i = sum_j Y_ij(s) * (V_j - V_ref).

  * DC / .op / .dc  : stamp the static conductance Y(0) directly.
  * AC              : nportacload.c stamps the complex Y(jw).
  * transient       : trapezoidal companion --
        - d_ij          -> constant conductance
        - e_ij * s      -> capacitor  I = e dV/dt         (trap companion)
        - res/(s - p)   -> first-order state  dx/dt = p x + u,  I += res x
                           (trap companion; x complex, conj pairs cancel to real)

The pole states x_jk depend only on the input j and pole k (shared across outputs
i), so they are updated exactly once per load (Phase A) and parked in CKTstate0;
Phase B recovers the history B_jk = x_jk - a_k*u_j from that parked value, so no
per-instance scratch is needed.
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "nportdefs.h"
#include "ngspice/sperror.h"

/* Y_ij(s) for s = sre + j*sim.  Shared with nportacload.c (DC and AC). */
void
NPORTadmittance(NPORTmodel *m, int i, int j, double sre, double sim,
                double *yre, double *yim)
{
    int N = m->NPORTnPorts, Np = m->NPORTnPoles, idx = i * N + j, k;
    double yr = m->NPORTd[idx] + sre * m->NPORTe[idx];
    double yi = sim * m->NPORTe[idx];

    for (k = 0; k < Np; k++) {
        double dre = sre - m->NPORTpoleRe[k];
        double dim = sim - m->NPORTpoleIm[k];
        double den = dre * dre + dim * dim;
        double rr = m->NPORTresRe[idx * Np + k];
        double ri = m->NPORTresIm[idx * Np + k];
        if (den == 0.0) continue;                 /* s exactly on a pole */
        yr += (rr * dre + ri * dim) / den;
        yi += (ri * dre - rr * dim) / den;
    }
    *yre = yr;
    *yim = yi;
}

/* trap coefficient a_k = (h/2) / (1 - (h/2) p_k)  (complex).  Depends on k, h. */
static void
nport_trap_a(double hh, double pr, double pi, double *ar, double *ai)
{
    double dr = 1.0 - hh * pr;     /* alpha = 1 - (h/2) p */
    double di =     - hh * pi;
    double mag = dr * dr + di * di;
    *ar = hh * dr / mag;           /* (h/2) / alpha */
    *ai = -hh * di / mag;
}

int
NPORTload(GENmodel *inModel, CKTcircuit *ckt)
{
    NPORTmodel *model = (NPORTmodel *)inModel;
    NPORTinstance *here;
    int i, j, k, N, Np;

    for (; model; model = NPORTnextModel(model)) {
        N  = model->NPORTnPorts;
        Np = model->NPORTnPoles;

        for (here = NPORTinstances(model); here; here = NPORTnextInstance(here)) {

            /* ---- DC / operating point : static admittance Y(0) ---- */
            if (ckt->CKTmode & MODEDC) {
                for (i = 0; i < N; i++)
                    for (j = 0; j < N; j++) {
                        double yr, yi;
                        NPORTadmittance(model, i, j, 0.0, 0.0, &yr, &yi);
                        *(here->NPORTyPtr[i * N + j]) += yr;
                        *(here->NPORTyColPtr[i])      += -yr;
                        *(here->NPORTyRowPtr[j])      += -yr;
                        *(here->NPORTyRefPtr)         += yr;
                    }
                continue;
            }

            /* ---- transient : trapezoidal companion ---- */
            {
                double  h      = ckt->CKTdelta;
                double  hh     = 0.5 * h;
                int    *node   = GENnode(&here->gen);
                int     ref    = here->NPORTrefNode;
                int     pBase  = here->NPORTstateBase;      /* poles: 4 per (j,k) */
                int     eBase  = pBase + 4 * N * Np;        /* e-cap: 2 per (i,j) */
                int     initTr = (ckt->CKTmode & MODEINITTRAN);
                int     uic    = (ckt->CKTmode & MODEUIC);
                double *st0    = ckt->CKTstate0;
                double *st1    = ckt->CKTstate1;
                double *rhsOld = ckt->CKTrhsOld;

                /* ---- Phase A: advance the shared pole states x_jk ---- */
                for (j = 0; j < N; j++) {
                    double uj = rhsOld[node[j]] - rhsOld[ref];
                    for (k = 0; k < Np; k++) {
                        double pr = model->NPORTpoleRe[k];
                        double pi = model->NPORTpoleIm[k];
                        int    s  = pBase + 4 * (j * Np + k);
                        double ar, ai, xr, xi, dxr, dxi, br, bi, mag;

                        nport_trap_a(hh, pr, pi, &ar, &ai);

                        if (initTr) {
                            if (uic) {          /* zero initial state */
                                xr = xi = 0.0;
                            } else {            /* DC steady state x = -u/p */
                                mag = pr * pr + pi * pi;
                                if (mag == 0.0) { xr = xi = 0.0; }
                                else { xr = -uj * pr / mag; xi =  uj * pi / mag; }
                            }
                            dxr = dxi = 0.0;    /* steady: dx/dt = 0 */
                        } else {
                            xr  = st1[s + 0];  xi  = st1[s + 1];
                            dxr = st1[s + 2];  dxi = st1[s + 3];
                        }

                        /* B = (x_n + (h/2) dx_n) / alpha,  alpha = 1 - (h/2) p */
                        {
                            double nr = xr + hh * dxr;
                            double ni = xi + hh * dxi;
                            double dr = 1.0 - hh * pr;
                            double di =     - hh * pi;
                            double dm = dr * dr + di * di;
                            br = (nr * dr + ni * di) / dm;
                            bi = (ni * dr - nr * di) / dm;
                        }
                        /* x_{n+1} = a*u_j + B   (u_j real) */
                        {
                            double xnr = ar * uj + br;
                            double xni = ai * uj + bi;
                            /* dx_{n+1} = p*x_{n+1} + u_j */
                            double dnr = pr * xnr - pi * xni + uj;
                            double dni = pr * xni + pi * xnr;
                            st0[s + 0] = xnr;  st0[s + 1] = xni;
                            st0[s + 2] = dnr;  st0[s + 3] = dni;
                        }
                    }
                }

                /* ---- Phase B: stamp conductance + history for every (i,j) ---- */
                for (i = 0; i < N; i++) {
                    for (j = 0; j < N; j++) {
                        int    idx  = i * N + j;
                        double uj   = rhsOld[node[j]] - rhsOld[ref];
                        double geq  = model->NPORTd[idx];   /* constant term */
                        double hist = 0.0;

                        /* pole contributions: recover B_jk = x_{n+1} - a_k*u_j */
                        for (k = 0; k < Np; k++) {
                            double pr = model->NPORTpoleRe[k];
                            double pi = model->NPORTpoleIm[k];
                            int    s  = pBase + 4 * (j * Np + k);
                            double ar, ai, br, bi, rr, ri;

                            nport_trap_a(hh, pr, pi, &ar, &ai);
                            br = st0[s + 0] - ar * uj;
                            bi = st0[s + 1] - ai * uj;
                            rr = model->NPORTresRe[idx * Np + k];
                            ri = model->NPORTresIm[idx * Np + k];
                            /* Re[res * a] adds to conductance; Re[res * B] to hist */
                            geq  += rr * ar - ri * ai;
                            hist += rr * br - ri * bi;
                        }

                        /* e-term capacitor: I = e dU/dt  (trapezoidal) */
                        {
                            double e = model->NPORTe[idx];
                            if (e != 0.0) {
                                int    es = eBase + 2 * idx;
                                double uPrev, iPrev, geqE, iNew, ieq;
                                if (initTr) {
                                    uPrev = uic ? 0.0 : uj;
                                    iPrev = 0.0;
                                } else {
                                    uPrev = st1[es + 0];
                                    iPrev = st1[es + 1];
                                }
                                geqE = 2.0 * e / h;
                                iNew = geqE * (uj - uPrev) - iPrev;
                                ieq  = -(geqE * uPrev + iPrev);
                                st0[es + 0] = uj;
                                st0[es + 1] = iNew;
                                geq  += geqE;
                                hist += ieq;
                            }
                        }

                        /* stamp four-corner conductance */
                        *(here->NPORTyPtr[idx])  += geq;
                        *(here->NPORTyColPtr[i]) += -geq;
                        *(here->NPORTyRowPtr[j]) += -geq;
                        *(here->NPORTyRefPtr)    += geq;

                        /* stamp equivalent-current history (ceq convention) */
                        ckt->CKTrhs[node[i]] -= hist;
                        ckt->CKTrhs[ref]     += hist;
                    }
                }
            }
        }
    }
    return OK;
}
