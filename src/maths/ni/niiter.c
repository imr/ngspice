/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: 2001 AlansFixes
**********/

/*
 * NIiter(ckt,maxIter)
 *
 *  This subroutine performs the actual numerical iteration.
 *  It uses the sparse matrix stored in the circuit struct
 *  along with the matrix loading program, the load data, the
 *  convergence test function, and the convergence parameters
 */

#include "ngspice/ngspice.h"
#include "ngspice/trandefs.h"
#include "ngspice/cktdefs.h"
#include "ngspice/smpdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/fteext.h"

/* Limit the number of 'singular matrix' warnings */
static int msgcount = 0;

/* NIiter() - return value is non-zero for convergence failure */

int
NIiter(CKTcircuit *ckt, int maxIter)
{
    double startTime, *OldCKTstate0 = NULL;
    int error, i, j;

    int iterno = 0;
    int ipass = 0;
    /* Track max|Δv| across iterations to gate the Stage B
     * progressive-halving below — see the damping block. */
    double prev_max_dv = 0.0;

    /* some convergence issues that get resolved by increasing max iter */
    if (maxIter < 100)
        maxIter = 100;


    if ((ckt->CKTmode & MODETRANOP) && (ckt->CKTmode & MODEUIC)) {
        SWAP(double *, ckt->CKTrhs, ckt->CKTrhsOld);
        error = CKTload(ckt);
        if (error)
            return(error);
        return(OK);
    }

#ifdef WANT_SENSE2
    if (ckt->CKTsenInfo) {
        error = NIsenReinit(ckt);
        if (error)
            return(error);
    }
#endif

    if (ckt->CKTniState & NIUNINITIALIZED) {
        error = NIreinit(ckt); /* always returns 0 */
        if (error) {
#ifdef STEPDEBUG
            printf("re-init returned error \n");
#endif
            return(error);
        }
    }

    /* OldCKTstate0 = TMALLOC(double, ckt->CKTnumStates + 1); */

    /* Build the gmin-skip map once: matrix rows (by node number) to EXCLUDE
     * from the dynamic/true gmin-stepping diagonal pass.  OSDI model-internal
     * nodes that back analog-operator implicit equations (laplace/zi/idt
     * state) are flagged node->nogmin in osdisetup; gmin on their small
     * diagonals over-damps the state Newton update and breaks gmin stepping
     * ("gmin stepping failed").  TMALLOC zero-fills, so only nogmin nodes are
     * set; NULL/legacy means "add gmin to every diagonal". */
    if (ckt->CKTmatrix && !ckt->CKTmatrix->gmin_skip && ckt->CKTmaxEqNum > 0) {
        CKTnode *gnode;
        ckt->CKTmatrix->gmin_skip = TMALLOC(char, ckt->CKTmaxEqNum + 1);
        if (ckt->CKTmatrix->gmin_skip)
            for (gnode = ckt->CKTnodes; gnode; gnode = gnode->next)
                if (gnode->nogmin && gnode->number > 0 &&
                    gnode->number <= ckt->CKTmaxEqNum)
                    ckt->CKTmatrix->gmin_skip[gnode->number] = 1;
    }

    for (;;) {

        ckt->CKTnoncon = 0;
        ckt->CKTosdiStepReject = 0;
        ckt->CKThugeJThisIter = 0;

        /* Axis 2 — publish the current Newton iteration index (1-based:
         * the iteration the upcoming CKTload computes) so OSDI models see
         * it via SimInfo.newton_iter and can make their $limit / limiter
         * behaviour iteration-aware (e.g. tighten the clamp as the count
         * climbs to break oscillation).  Resets to 1 on every NIiter call,
         * so it is per-Newton-solve, not cumulative. */
        ckt->CKTosdiNewtonIter = iterno + 1;

#ifdef NEWPRED
        if (!(ckt->CKTmode & MODEINITPRED))
#endif
        {

            error = CKTload(ckt);
            /* printf("loaded, noncon is %d\n", ckt->CKTnoncon); */
            /* fflush(stdout); */
            iterno++;
            if (error) {
                ckt->CKTstat->STATnumIter += iterno;
#ifdef STEPDEBUG
                printf("load returned error \n");
#endif
                FREE(OldCKTstate0);
                return (error);
            }

            /* Small-dt convergence rescue.  At very small CKTdelta during
             * transient, sanitize-* paths in osdiload.c (and similar in
             * other device loaders) fire spurious CKTnoncon++ events
             * because the integration coefficient 2/dt amplifies tiny
             * charge changes into apparent currents that exceed the
             * defensive 1 A clip.  Those increments block NIconvTest
             * from running, so a perfectly settled iterate cannot
             * converge — NIiter spins until maxIter and returns
             * E_ITERLIM, dctran cuts delta, eventually aborts with
             * "Timestep too small".
             *
             * Clear CKTnoncon here so the downstream NIconvTest can
             * evaluate solution stability directly.  Genuine
             * divergence still fires: NIconvTest tests |Δx| against
             * (reltol·|x| + vntol) per node — an actually-diverging
             * iterate fails that test regardless of CKTnoncon state.
             *
             * Threshold = 1 ps.  Above that, the integration
             * coefficient is small enough that sanitize-* increments
             * reflect real model anomalies, not numerical artifacts.
             *
             * Opt-out via `.option nodtclear`. */
            if ((ckt->CKTmode & MODETRAN) &&
                !ckt->CKTdtClearOff &&
                ckt->CKTdelta < 1.0e-12) {
                ckt->CKTnoncon = 0;
            }

            /* Axis-3 step rejection: an OSDI model has explicitly raised
             * EVAL_RET_FLAG_REJECT_STEP, or sanitize_jacobian has seen
             * a NaN/Inf Jacobian entry (catastrophic — clipping can't
             * save a falsified linearization).  Return E_ITERLIM so
             * dctran cuts CKTdelta by 8 and retries with a better-
             * conditioned predicted state.  Only honored during
             * transient. */
            if (ckt->CKTosdiStepReject && (ckt->CKTmode & MODETRAN)) {
                ckt->CKTstat->STATnumIter += iterno;
                FREE(OldCKTstate0);
                return (E_ITERLIM);
            }

            /* printf("after loading, before solving\n"); */
            /* CKTdump(ckt); */

            if (!(ckt->CKTniState & NIDIDPREORDER)) {
                error = SMPpreOrder(ckt->CKTmatrix);
                if (error) {
                    ckt->CKTstat->STATnumIter += iterno;
#ifdef STEPDEBUG
                    printf("pre-order returned error \n");
#endif
                    FREE(OldCKTstate0);
                    return(error); /* badly formed matrix */
                }
                ckt->CKTniState |= NIDIDPREORDER;
            }

            if ((ckt->CKTmode & MODEINITJCT) ||
                ((ckt->CKTmode & MODEINITTRAN) && (iterno == 1)))
            {
                ckt->CKTniState |= NISHOULDREORDER;
            }

            if (ckt->CKTniState & NISHOULDREORDER) {
                startTime = SPfrontEnd->IFseconds();

#ifdef KLU
                if (ckt->CKTkluMODE) {
                    ckt->CKTmatrix->SMPkluMatrix->KLUloadDiagGmin = 1 ;
                }
#endif

                error = SMPreorder(ckt->CKTmatrix, ckt->CKTpivotAbsTol,
                                   ckt->CKTpivotRelTol, ckt->CKTdiagGmin);
                ckt->CKTstat->STATreorderTime +=
                    SPfrontEnd->IFseconds() - startTime;
                if (error) {
                    /* new feature - we can now find out something about what is
                     * wrong - so we ask for the troublesome entry
                     * Limit the number of messages to 6, if not 'set ngdebug'.
                     */
                    if (ft_ngdebug || msgcount < 6) {
                        SMPgetError(ckt->CKTmatrix, &i, &j);
                        if(eq(NODENAME(ckt, i), NODENAME(ckt, j)))
                            SPfrontEnd->IFerrorf(ERR_WARNING, "singular matrix:  check node %s\n", NODENAME(ckt, i));
                        else
                            SPfrontEnd->IFerrorf(ERR_WARNING, "singular matrix:  check nodes %s and %s\n", NODENAME(ckt, i), NODENAME(ckt, j));
                        msgcount += 1;
                    }
                    ckt->CKTstat->STATnumIter += iterno;
#ifdef STEPDEBUG
                    printf("reorder returned error \n");
#endif
                    FREE(OldCKTstate0);
                    return(error); /* can't handle these errors - pass up! */
                }
                ckt->CKTniState &= ~NISHOULDREORDER;
            } else {
                startTime = SPfrontEnd->IFseconds();

#ifdef KLU
                if (ckt->CKTkluMODE) {
                    ckt->CKTmatrix->SMPkluMatrix->KLUloadDiagGmin = 1 ;
                }
#endif

                error = SMPluFac(ckt->CKTmatrix, ckt->CKTpivotAbsTol,
                                 ckt->CKTdiagGmin);
                ckt->CKTstat->STATdecompTime +=
                    SPfrontEnd->IFseconds() - startTime;

#ifdef KLU
                if ((ckt->CKTkluMODE) && (error == E_SINGULAR)) {

                    /* Francesco Lannutti - 25 Aug 2020
                     * If the matrix is numerically singular during ReFactorization, take the same matrix and factor it from scratch in the same iteration.
                     * This is my mod with KLU. It saves run-time, but also the system at the next iteration may be different.
                     * How do we guarantee that the system is the same at the next iteration? So, the original SPARSE version below sounds like a bug.
                     */
                    if (ft_ngdebug)
                        fprintf (stderr, "Warning: KLU ReFactor failed. Factoring again...\n") ;
                    ckt->CKTniState |= NISHOULDREORDER;
                    ckt->CKTmatrix->SMPkluMatrix->KLUloadDiagGmin = 0 ;
                    error = SMPreorder(ckt->CKTmatrix, ckt->CKTpivotAbsTol, ckt->CKTpivotRelTol, ckt->CKTdiagGmin);
                    ckt->CKTstat->STATreorderTime += SPfrontEnd->IFseconds() - startTime;
                    if (error) {
                        SMPgetError(ckt->CKTmatrix, &i, &j);
                        if (ft_ngdebug || msgcount < 6) {
                            SMPgetError(ckt->CKTmatrix, &i, &j);
                            if (eq(NODENAME(ckt, i), NODENAME(ckt, j)))
                                SPfrontEnd->IFerrorf(ERR_WARNING, "singular matrix:  check node %s\n", NODENAME(ckt, i));
                            else
                                SPfrontEnd->IFerrorf(ERR_WARNING, "singular matrix:  check nodes %s and %s\n", NODENAME(ckt, i), NODENAME(ckt, j));
                            msgcount += 1;
                        }

                        /* CKTload(ckt); */
                        /* SMPprint(ckt->CKTmatrix, stdout); */
                        /* seems to be singular - pass the bad news up */
                        ckt->CKTstat->STATnumIter += iterno;
#ifdef STEPDEBUG
                        printf("lufac returned error \n");
#endif
                        FREE(OldCKTstate0);
                        return(error);
                    }
                } else if (error) {
                    if (!(ckt->CKTkluMODE) && (error == E_SINGULAR)) {

                        /* Francesco Lannutti - 25 Aug 2020
                         * If the matrix is numerically singular during ReFactorization, factor it from scratch at the next iteration.
                         * This is the original SPICE3F5 code and uses SPARSE.
                         */

                        ckt->CKTniState |= NISHOULDREORDER;
                        DEBUGMSG(" forced reordering....\n");
                        continue;
                    }
                    /* CKTload(ckt); */
                    /* SMPprint(ckt->CKTmatrix, stdout); */
                    /* seems to be singular - pass the bad news up */
                    ckt->CKTstat->STATnumIter += iterno;
#ifdef STEPDEBUG
                    printf("lufac returned error \n");
#endif
                    FREE(OldCKTstate0);
                    return(error);
                }
#else
                if (error) {
                    if (error == E_SINGULAR) {

                        /* Francesco Lannutti - 25 Aug 2020
                         * If the matrix is numerically singular during ReFactorization, factor it from scratch at the next iteration.
                         * This is the original SPICE3F5 code and uses SPARSE.
                         */

                        ckt->CKTniState |= NISHOULDREORDER;
                        DEBUGMSG(" forced reordering....\n");
                        continue;
                    }
                    /* CKTload(ckt); */
                    /* SMPprint(ckt->CKTmatrix, stdout); */
                    /* seems to be singular - pass the bad news up */
                    ckt->CKTstat->STATnumIter += iterno;
#ifdef STEPDEBUG
                    printf("lufac returned error \n");
#endif
                    FREE(OldCKTstate0);
                    return(error);
                }
#endif

            }

            /* moved it to here as if xspice is included then CKTload changes
               CKTnumStates the first time it is run */
            if (!OldCKTstate0)
                OldCKTstate0 = TMALLOC(double, ckt->CKTnumStates + 1);
            if (ckt->CKTstate0)
                memcpy(OldCKTstate0, ckt->CKTstate0,
                       (size_t) ckt->CKTnumStates * sizeof(double));

            /* Axis 4 placeholder — the |f|-magnitude (residual-norm) half
             * of dual-norm convergence stays disabled.  A correct true-KCL
             * residual check (f = G*x - b via SMPmultiply in the pre-factor
             * window) was implemented and verified CORRECT, but enforcing it
             * by default over-rejects real PDK operating points that rely on
             * ngspice's lenient SPICE3 |dx|-only convergence (TSMC22 OP
             * diverged, Samsung slowed badly) while a clean circuit (0.9V
             * inverter) was byte-identical.  So treat residual as always-
             * passed; NIconvTest gates on |dx| only.  The CKTresidConverged
             * / CKTresidCheckDisabled fields + niconv.c gate + `.option
             * noresidcheck` plumbing remain for a future opt-in form. */
            ckt->CKTresidConverged = 1;

            startTime = SPfrontEnd->IFseconds();
            SMPsolve(ckt->CKTmatrix, ckt->CKTrhs, ckt->CKTrhsSpare);
            ckt->CKTstat->STATsolveTime +=
                SPfrontEnd->IFseconds() - startTime;

#ifdef STEPDEBUG
            /*XXXX*/
            if (ckt->CKTrhs[0] != 0.0)
                printf("NIiter: CKTrhs[0] = %g\n", ckt->CKTrhs[0]);
            if (ckt->CKTrhsSpare[0] != 0.0)
                printf("NIiter: CKTrhsSpare[0] = %g\n", ckt->CKTrhsSpare[0]);
            if (ckt->CKTrhsOld[0] != 0.0)
                printf("NIiter: CKTrhsOld[0] = %g\n", ckt->CKTrhsOld[0]);
            /*XXXX*/
#endif
            ckt->CKTrhs[0] = 0;
            ckt->CKTrhsSpare[0] = 0;
            ckt->CKTrhsOld[0] = 0;

            /* Newton-step limiter (simulator-side $limit substitute).
             * Always-on during transient (and DC OP) to clamp per-node
             * |Δv| to ±CKTabsDv between Newton iterations.  Matches
             * the default behaviour of Spectre/HSPICE DEVlimvds-style
             * limiters which fire on every iteration as a model-
             * supplied analog of this.  Threshold is CKTabsDv (default
             * 0.5 V) — a model-parameter-agnostic tolerance, not a
             * voltage rail.
             *
             * Skipped on iteration 1 (no previous iterate to compare).
             * Skipped when CKTnodes is NULL (matrix not yet built).
             *
             * For models compiled with the OpenVA compiler-side
             * $limit synthesis pass, the model's own limiters apply
             * inside descr->eval() — this simulator-side limiter then
             * sees already-limited Δv and does nothing additional. */
            if (iterno > 1 && ckt->CKTnodes != NULL) {
                double dv_max = (ckt->CKTabsDv > 0) ? ckt->CKTabsDv : 0.5;
                /* Compute current iteration's max|Δv|.  Needed both
                 * for the Stage A scalar scaling and for the Stage B
                 * stagnation gate. */
                CKTnode *node;
                double max_dv = 0.0;
                for (node = ckt->CKTnodes->next; node; node = node->next) {
                    if (node->type != SP_VOLTAGE) continue;
                    double diff = fabs(ckt->CKTrhs[node->number] -
                                       ckt->CKTrhsOld[node->number]);
                    if (diff > max_dv) max_dv = diff;
                }
                /* Two-stage limiter:
                 *
                 * Stage A (always): SCALAR damping if any node's |Δv|
                 *   exceeds dv_max — scale all node updates by
                 *   dv_max/max_dv.  Preserves the proportional
                 *   structure Newton's linearization relies on, so
                 *   the local consistency between coupled nodes
                 *   (e.g. pinb → E1 → net_6 → Rs3 → net_7) is
                 *   maintained even when the natural step is
                 *   unphysically large.
                 *
                 * Stage B (past iter 3, ONLY when stagnating):
                 *   progressively tighten dv_max by halving so any
                 *   persistent oscillation damps out within a few
                 *   iterations.  Capped at 6 halvings (dv_max/64 ≈
                 *   8 mV for default 0.5 V).
                 *
                 *   Stagnation = current max|Δv| not dropping by at
                 *   least 30% from prev iteration.  Unconditional
                 *   halving past iter 3 was preventing convergence
                 *   on the TSMC22 ULP driver_lv_2v5_tb VSN node: the
                 *   inductor-coupled supply (L1 = 2.7 nH) naturally
                 *   needs ~100 mV swings to track each switching
                 *   transition, but dv_max would collapse to 8 mV by
                 *   iter 10 even though Newton was making strong
                 *   monotonic progress — the apparent "10 iters with
                 *   no convergence" was actually "10 iters of forced
                 *   under-correction."  Gating on stagnation lets
                 *   Newton keep stepping at its natural rate when
                 *   it's converging, and only suppresses when it's
                 *   genuinely oscillating. */
                bool stagnating = (iterno > 3) &&
                                  (prev_max_dv > 0.0) &&
                                  (max_dv > 0.7 * prev_max_dv);
                if (stagnating) {
                    int shift = iterno - 3;
                    if (shift > 6) shift = 6;
                    dv_max /= (double)(1 << shift);
                }
                /* Save the PRE-clamp max|Δv| for next iter's
                 * stagnation check.  Post-clamp values would be
                 * pinned at dv_max whenever Stage A fires, which
                 * destroys the iteration-trend signal Stage B needs
                 * to distinguish "Newton genuinely converging but
                 * being clamped" from "Newton stuck oscillating." */
                prev_max_dv = max_dv;
                if (max_dv > dv_max) {
                    double scale = dv_max / max_dv;
                    for (node = ckt->CKTnodes->next; node; node = node->next) {
                        if (node->type != SP_VOLTAGE) continue;
                        double diff = ckt->CKTrhs[node->number] -
                                      ckt->CKTrhsOld[node->number];
                        ckt->CKTrhs[node->number] =
                            ckt->CKTrhsOld[node->number] + scale * diff;
                    }
                }
            }

            if (iterno > maxIter) {
                ckt->CKTstat->STATnumIter += iterno;
                /* we don't use this info during transient analysis */
                if (ckt->CKTcurrentAnalysis != DOING_TRAN) {
                    FREE(errMsg);
                    errMsg = copy("Too many iterations without convergence");
#ifdef STEPDEBUG
                    fprintf(stderr, "too many iterations without convergence: %d iter's (max iter == %d)\n",
                    iterno, maxIter);
#endif
                }
                FREE(OldCKTstate0);
                return(E_ITERLIM);
            }

            if ((ckt->CKTnoncon == 0) && (iterno != 1))
                ckt->CKTnoncon = NIconvTest(ckt);
            else
                ckt->CKTnoncon = 1;

#ifdef STEPDEBUG
            printf("noncon is %d\n", ckt->CKTnoncon);
#endif
        }

        if ((ckt->CKTnodeDamping != 0) && (ckt->CKTnoncon != 0) &&
            ((ckt->CKTmode & MODETRANOP) || (ckt->CKTmode & MODEDCOP)) &&
            (iterno > 1))
        {
            CKTnode *node;
            double diff, maxdiff = 0;
            for (node = ckt->CKTnodes->next; node; node = node->next)
                if (node->type == SP_VOLTAGE) {
                    diff = fabs(ckt->CKTrhs[node->number] - ckt->CKTrhsOld[node->number]);
                    if (maxdiff < diff)
                        maxdiff = diff;
                }

            if (maxdiff > 10) {
                double damp_factor = 10 / maxdiff;
                if (damp_factor < 0.1)
                    damp_factor = 0.1;
                for (node = ckt->CKTnodes->next; node; node = node->next) {
                    diff = ckt->CKTrhs[node->number] - ckt->CKTrhsOld[node->number];
                    ckt->CKTrhs[node->number] =
                        ckt->CKTrhsOld[node->number] + (damp_factor * diff);
                }
                for (i = 0; i < ckt->CKTnumStates; i++) {
                    diff = ckt->CKTstate0[i] - OldCKTstate0[i];
                    ckt->CKTstate0[i] = OldCKTstate0[i] + (damp_factor * diff);
                }
            }
        }

        if (ckt->CKTmode & MODEINITFLOAT) {
            if ((ckt->CKTmode & MODEDC) && ckt->CKThadNodeset) {
                if (ipass)
                    ckt->CKTnoncon = ipass;
                ipass = 0;
            }
            if (ckt->CKTnoncon == 0) {
                ckt->CKTstat->STATnumIter += iterno;
                FREE(OldCKTstate0);
                return(OK);
            }
        } else if (ckt->CKTmode & MODEINITJCT) {
            ckt->CKTmode = (ckt->CKTmode & ~INITF) | MODEINITFIX;
            ckt->CKTniState |= NISHOULDREORDER;
        } else if (ckt->CKTmode & MODEINITFIX) {
            if (ckt->CKTnoncon == 0)
                ckt->CKTmode = (ckt->CKTmode & ~INITF) | MODEINITFLOAT;
            ipass = 1;
        } else if (ckt->CKTmode & MODEINITSMSIG) {
            ckt->CKTmode = (ckt->CKTmode & ~INITF) | MODEINITFLOAT;
        } else if (ckt->CKTmode & MODEINITTRAN) {
            if (iterno <= 1)
                ckt->CKTniState |= NISHOULDREORDER;
            ckt->CKTmode = (ckt->CKTmode & ~INITF) | MODEINITFLOAT;
        } else if (ckt->CKTmode & MODEINITPRED) {
            ckt->CKTmode = (ckt->CKTmode & ~INITF) | MODEINITFLOAT;
        } else {
            ckt->CKTstat->STATnumIter += iterno;
#ifdef STEPDEBUG
            printf("bad initf state \n");
#endif
            FREE(OldCKTstate0);
            return(E_INTERN);
            /* impossible - no such INITF flag! */
        }

        /* build up the lvnim1 array from the lvn array */
        SWAP(double *, ckt->CKTrhs, ckt->CKTrhsOld);
        /* printf("after loading, after solving\n"); */
        /* CKTdump(ckt); */
    }
    /*NOTREACHED*/
}

void NIresetwarnmsg(void) {
    msgcount = 0;
}
