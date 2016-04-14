/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: 2000  AlansFixes
Modified: 2005 Paolo Nenzi - Restructured
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/devdefs.h"
#include "ngspice/sperror.h"

#ifdef XSPICE
#include "ngspice/enh.h"
#endif


static int dynamic_gmin (CKTcircuit *, long int, long int, int);
static int spice3_gmin (CKTcircuit *, long int, long int, int);
static int gillespie_src (CKTcircuit *, long int, long int, int);
static int spice3_src (CKTcircuit *, long int, long int, int);


int
CKTop (CKTcircuit * ckt, long int firstmode, long int continuemode,
       int iterlim)
{
    int converged;
#ifdef HAS_PROGREP
    SetAnalyse("op", 0);
#endif
    ckt->CKTmode = firstmode;

    if (!ckt->CKTnoOpIter) {
#ifdef XSPICE
        /* gtri - begin - wbk - add convergence problem reporting flags */
        if ((ckt->CKTnumGminSteps <= 0) && (ckt->CKTnumSrcSteps <= 0))
            ckt->enh->conv_debug.last_NIiter_call = MIF_TRUE;
        else
            ckt->enh->conv_debug.last_NIiter_call = MIF_FALSE;
        /* gtri - end - wbk - add convergence problem reporting flags */
#endif
        converged = NIiter (ckt, iterlim);
    } else {
        converged = 1;          /* the 'go directly to gmin stepping' option */
    }


    if (converged != 0) {
        /* no convergence on the first try, so we do something else */
        /* first, check if we should try gmin stepping */

        if (ckt->CKTnumGminSteps >= 1) {
            if (ckt->CKTnumGminSteps == 1)
                converged = dynamic_gmin(ckt, firstmode, continuemode, iterlim);
            else
                converged = spice3_gmin(ckt, firstmode, continuemode, iterlim);
        }
        if (!converged)         /* If gmin-stepping worked... move out */
            return (0);

        /* ... otherwise try stepping sources ...
         * now, we'll try source stepping - we scale the sources
         * to 0, converge, then start stepping them up until they
         * are at their normal values
         */

        if (ckt->CKTnumSrcSteps >= 1) {
            if (ckt->CKTnumSrcSteps == 1)
                converged = gillespie_src(ckt, firstmode, continuemode, iterlim);
            else
                converged = spice3_src(ckt, firstmode, continuemode, iterlim);
        }
#ifdef XSPICE
        /* gtri - begin - wbk - add convergence problem reporting flags */
        ckt->enh->conv_debug.last_NIiter_call = MIF_FALSE;
        /* gtri - end - wbk - add convergence problem reporting flags */
#endif

    }

    return (converged);
}



/* CKTconvTest(ckt)
 *    this is a driver program to iterate through all the various
 *    convTest functions provided for the circuit elements in the
 *    given circuit
 */

int
CKTconvTest (CKTcircuit * ckt)
{
    int i;
    int error = OK;

    for (i = 0; i < DEVmaxnum; i++) {
        if (DEVices[i] && DEVices[i]->DEVconvTest && ckt->CKThead[i]) {
            error = DEVices[i]->DEVconvTest (ckt->CKThead[i], ckt);
        }

        if (error)
            return (error);

        if (ckt->CKTnoncon) {
            /* printf("convTest: device %s failed\n",
             * DEVices[i]->DEVpublic.name); */
            return (OK);
        }
    }

    return (OK);
}


/* Dynamic gmin stepping
 * Algorithm by Alan Gillespie
 * Modified 2005 - Paolo Nenzi (extracted from CKTop.c code)
 *
 * return value:
 * 0 -> method converged
 * 1 -> method failed
 *
 * Note that no path out of this code allows ckt->CKTdiagGmin to be
 * anything but CKTgshunt.
 */

static int
dynamic_gmin (CKTcircuit * ckt, long int firstmode,
              long int continuemode, int iterlim)
{
    double OldGmin, gtarget, factor;
    int success, failed, converged;

    int NumNodes, iters, i;
    double *OldRhsOld, *OldCKTstate0;
    CKTnode *n;

    ckt->CKTmode = firstmode;
    SPfrontEnd->IFerrorf (ERR_INFO,
                         "Starting dynamic gmin stepping");

    NumNodes = 0;
    for (n = ckt->CKTnodes; n; n = n->next)
        NumNodes++;

    OldRhsOld = TMALLOC(double, NumNodes + 1);
    OldCKTstate0 =
        TMALLOC(double, ckt->CKTnumStates + 1);

    for (n = ckt->CKTnodes; n; n = n->next)
        ckt->CKTrhsOld [n->number] = 0;

    for (i = 0; i < ckt->CKTnumStates; i++)
        ckt->CKTstate0 [i] = 0;

    factor = ckt->CKTgminFactor;
    OldGmin = 1e-2;
    ckt->CKTdiagGmin = OldGmin / factor;
    gtarget = MAX (ckt->CKTgmin, ckt->CKTgshunt);
    success = failed = 0;

    while ((!success) && (!failed)) {
        fprintf (stderr, "Trying gmin = %12.4E ", ckt->CKTdiagGmin);
        ckt->CKTnoncon = 1;
        iters = ckt->CKTstat->STATnumIter;

        converged = NIiter (ckt, ckt->CKTdcTrcvMaxIter);
        iters = (ckt->CKTstat->STATnumIter) - iters;

        if (converged == 0) {
            ckt->CKTmode = continuemode;
            SPfrontEnd->IFerrorf (ERR_INFO,
                                 "One successful gmin step");

            if (ckt->CKTdiagGmin <= gtarget) {
                success = 1;
            } else {
                i = 0;
                for (n = ckt->CKTnodes; n; n = n->next) {
                    OldRhsOld[i] = ckt->CKTrhsOld[n->number];
                    i++;
                }

                for (i = 0; i < ckt->CKTnumStates; i++) {
                    OldCKTstate0[i] = ckt->CKTstate0[i];
                }

                if (iters <= (ckt->CKTdcTrcvMaxIter / 4)) {
                    factor *= sqrt (factor);
                    if (factor > ckt->CKTgminFactor)
                        factor = ckt->CKTgminFactor;
                }

                if (iters > (3 * ckt->CKTdcTrcvMaxIter / 4))
                    factor = sqrt (factor);

                OldGmin = ckt->CKTdiagGmin;

                if ((ckt->CKTdiagGmin) < (factor * gtarget)) {
                    factor = ckt->CKTdiagGmin / gtarget;
                    ckt->CKTdiagGmin = gtarget;
                } else {
                    ckt->CKTdiagGmin /= factor;
                }
            }
        } else {
            if (factor < 1.00005) {
                failed = 1;
                SPfrontEnd->IFerrorf (ERR_WARNING,
                                     "Last gmin step failed");
            } else {
                SPfrontEnd->IFerrorf (ERR_WARNING,
                                     "Further gmin increment");
                factor = sqrt (sqrt (factor));
                ckt->CKTdiagGmin = OldGmin / factor;

                i = 0;
                for (n = ckt->CKTnodes; n; n = n->next) {
                    ckt->CKTrhsOld[n->number] = OldRhsOld[i];
                    i++;
                }

                for (i = 0; i < ckt->CKTnumStates; i++) {
                    ckt->CKTstate0[i] = OldCKTstate0[i];
                }
            }
        }
    }

    ckt->CKTdiagGmin = ckt->CKTgshunt;
    FREE (OldRhsOld);
    FREE (OldCKTstate0);

#ifdef XSPICE
    /* gtri - begin - wbk - add convergence problem reporting flags */
    if (ckt->CKTnumSrcSteps <= 0)
        ckt->enh->conv_debug.last_NIiter_call = MIF_TRUE;
    else
        ckt->enh->conv_debug.last_NIiter_call = MIF_FALSE;
    /* gtri - end - wbk - add convergence problem reporting flags */
#endif

    converged = NIiter (ckt, iterlim);

    if (converged != 0) {
        SPfrontEnd->IFerrorf (ERR_WARNING,
                             "Dynamic gmin stepping failed");
    } else {
        SPfrontEnd->IFerrorf (ERR_INFO,
                             "Dynamic gmin stepping completed");
#ifdef XSPICE
        /* gtri - begin - wbk - add convergence problem reporting flags */
        ckt->enh->conv_debug.last_NIiter_call = MIF_FALSE;
        /* gtri - end - wbk - add convergence problem reporting flags */
#endif
    }

    return (converged);
}


/* Spice3 gmin stepping
 * Modified 2000 - Alan Gillespie (added gshunt)
 * Modified 2005 - Paolo Nenzi (extracted from CKTop.c code)
 *
 * return value:
 * 0 -> method converged
 * 1 -> method failed
 *
 * Note that no path out of this code allows ckt->CKTdiagGmin to be
 * anything but CKTgshunt.
 */

static int
spice3_gmin (CKTcircuit * ckt, long int firstmode,
             long int continuemode, int iterlim)
{

    int converged, i;

    ckt->CKTmode = firstmode;
    SPfrontEnd->IFerrorf (ERR_INFO,
                         "Starting gmin stepping");

    if (ckt->CKTgshunt == 0)
        ckt->CKTdiagGmin = ckt->CKTgmin;
    else
        ckt->CKTdiagGmin = ckt->CKTgshunt;

    for (i = 0; i < ckt->CKTnumGminSteps; i++)
        ckt->CKTdiagGmin *= ckt->CKTgminFactor;


    for (i = 0; i <= ckt->CKTnumGminSteps; i++) {
        fprintf (stderr, "Trying gmin = %12.4E ", ckt->CKTdiagGmin);
        ckt->CKTnoncon = 1;
        converged = NIiter (ckt, ckt->CKTdcTrcvMaxIter);

        if (converged != 0) {
            ckt->CKTdiagGmin = ckt->CKTgshunt;
            SPfrontEnd->IFerrorf (ERR_WARNING,
                                 "gmin step failed");
            break;
        }

        ckt->CKTdiagGmin /= ckt->CKTgminFactor;
        ckt->CKTmode = continuemode;

        SPfrontEnd->IFerrorf (ERR_INFO,
                             "One successful gmin step");
    }

    ckt->CKTdiagGmin = ckt->CKTgshunt;

#ifdef XSPICE
    /* gtri - begin - wbk - add convergence problem reporting flags */
    if (ckt->CKTnumSrcSteps <= 0)
        ckt->enh->conv_debug.last_NIiter_call = MIF_TRUE;
    else
        ckt->enh->conv_debug.last_NIiter_call = MIF_FALSE;
    /* gtri - end - wbk - add convergence problem reporting flags */
#endif

    converged = NIiter (ckt, iterlim);

    if (converged == 0) {
        SPfrontEnd->IFerrorf (ERR_INFO,
                             "gmin stepping completed");

#ifdef XSPICE
        /* gtri - begin - wbk - add convergence problem reporting flags */
        ckt->enh->conv_debug.last_NIiter_call = MIF_FALSE;
        /* gtri - end - wbk - add convergence problem reporting flags */
#endif

    } else {
        SPfrontEnd->IFerrorf (ERR_WARNING,
                             "gmin stepping failed");
    }

    return (converged);
}


/* Gillespie's Source stepping
 * Modified 2005 - Paolo Nenzi (extracted from CKTop.c code)
 *
 * return value:
 * 0 -> method converged
 * 1 -> method failed
 *
 * Note that no path out of this code allows ckt->CKTsrcFact to be
 * anything but 1.00000.
 */
static int
gillespie_src (CKTcircuit * ckt, long int firstmode,
               long int continuemode, int iterlim)
{

    int converged, NumNodes, i, iters;
    double raise, ConvFact;
    double *OldRhsOld, *OldCKTstate0;
    CKTnode *n;

    NG_IGNORE(iterlim);

    ckt->CKTmode = firstmode;
    SPfrontEnd->IFerrorf (ERR_INFO,
                         "Starting source stepping");

    ckt->CKTsrcFact = 0;
    raise = 0.001;
    ConvFact = 0;

    NumNodes = 0;
    for (n = ckt->CKTnodes; n; n = n->next) {
        NumNodes++;
    }

    OldRhsOld = TMALLOC(double, NumNodes + 1);
    OldCKTstate0 =
        TMALLOC(double, ckt->CKTnumStates + 1);

    for (n = ckt->CKTnodes; n; n = n->next)
        ckt->CKTrhsOld[n->number] = 0;

    for (i = 0; i < ckt->CKTnumStates; i++)
        ckt->CKTstate0[i] = 0;

    /*  First, try a straight solution with all sources at zero */

    fprintf (stderr, "Supplies reduced to %8.4f%% ", ckt->CKTsrcFact * 100);
    converged = NIiter (ckt, ckt->CKTdcTrcvMaxIter);

    /*  If this doesn't work, try gmin stepping as well for the first solution */

    if (converged != 0) {
        fprintf (stderr, "\n");
        if (ckt->CKTgshunt <= 0) {
            ckt->CKTdiagGmin = ckt->CKTgmin;
        } else {
            ckt->CKTdiagGmin = ckt->CKTgshunt;
        }

        for (i = 0; i < 10; i++)
            ckt->CKTdiagGmin *= 10;

        for (i = 0; i <= 10; i++) {
            fprintf (stderr, "Trying gmin = %12.4E ", ckt->CKTdiagGmin);
            ckt->CKTnoncon = 1;

#ifdef XSPICE
            /* gtri - begin - wbk - add convergence problem reporting flags */
            ckt->enh->conv_debug.last_NIiter_call = MIF_TRUE;
            /* gtri - end - wbk - add convergence problem reporting flags */
#endif

            converged = NIiter (ckt, ckt->CKTdcTrcvMaxIter);

            if (converged != 0) {
                ckt->CKTdiagGmin = ckt->CKTgshunt;
                SPfrontEnd->IFerrorf (ERR_WARNING,
                                     "gmin step failed");
#ifdef XSPICE
                /* gtri - begin - wbk - add convergence problem reporting flags */
                ckt->enh->conv_debug.last_NIiter_call = MIF_FALSE;
                /* gtri - end - wbk - add convergence problem reporting flags */
#endif
                break;
            }

            ckt->CKTdiagGmin /= 10;
            ckt->CKTmode = continuemode;
            SPfrontEnd->IFerrorf (ERR_INFO,
                                 "One successful gmin step");
        }
        ckt->CKTdiagGmin = ckt->CKTgshunt;
    }

    /*  If we've got convergence, then try stepping up the sources  */

    if (converged == 0) {
        i = 0;
        for (n = ckt->CKTnodes; n; n = n->next) {
            OldRhsOld[i] = ckt->CKTrhsOld[n->number];
            i++;
        }

        for (i = 0; i < ckt->CKTnumStates; i++)
            OldCKTstate0[i] = ckt->CKTstate0[i];


        SPfrontEnd->IFerrorf (ERR_INFO,
                             "One successful source step");
        ckt->CKTsrcFact = ConvFact + raise;
    }


    if (converged == 0)
        do {
            fprintf (stderr,
                     "Supplies reduced to %8.4f%% ", ckt->CKTsrcFact * 100);

            iters = ckt->CKTstat->STATnumIter;

#ifdef XSPICE
            /* gtri - begin - wbk - add convergence problem reporting flags */
            ckt->enh->conv_debug.last_NIiter_call = MIF_TRUE;
            /* gtri - end - wbk - add convergence problem reporting flags */
#endif
            converged = NIiter (ckt, ckt->CKTdcTrcvMaxIter);

            iters = (ckt->CKTstat->STATnumIter) - iters;

            ckt->CKTmode = continuemode;

            if (converged == 0) {
                ConvFact = ckt->CKTsrcFact;
                i = 0;

                for (n = ckt->CKTnodes; n; n = n->next) {
                    OldRhsOld[i] = ckt->CKTrhsOld[n->number];
                    i++;
                }

                for (i = 0; i < ckt->CKTnumStates; i++)
                    OldCKTstate0[i] = ckt->CKTstate0[i];

                SPfrontEnd->IFerrorf (ERR_INFO,
                                     "One successful source step");

                ckt->CKTsrcFact = ConvFact + raise;

                if (iters <= (ckt->CKTdcTrcvMaxIter / 4)) {
                    raise = raise * 1.5;
                }

                if (iters > (3 * ckt->CKTdcTrcvMaxIter / 4)) {
                    raise = raise * 0.5;
                }

                /*                    if (raise>0.01) raise=0.01; */

            } else {

                if ((ckt->CKTsrcFact - ConvFact) < 1e-8)
                    break;

                raise = raise / 10;

                if (raise > 0.01)
                    raise = 0.01;

                ckt->CKTsrcFact = ConvFact;
                i = 0;

                for (n = ckt->CKTnodes; n; n = n->next) {
                    ckt->CKTrhsOld[n->number] = OldRhsOld[i];
                    i++;
                }

                for (i = 0; i < ckt->CKTnumStates; i++)
                    ckt->CKTstate0[i] = OldCKTstate0[i];

            }

            if ((ckt->CKTsrcFact) > 1)
                ckt->CKTsrcFact = 1;

        } while ((raise >= 1e-7) && (ConvFact < 1));

    FREE (OldRhsOld);
    FREE (OldCKTstate0);
    ckt->CKTsrcFact = 1;

    if (ConvFact != 1) {
        ckt->CKTsrcFact = 1;
        ckt->CKTcurrentAnalysis = DOING_TRAN;
        SPfrontEnd->IFerrorf (ERR_WARNING,
                             "source stepping failed");
        return (E_ITERLIM);
    } else {
        SPfrontEnd->IFerrorf (ERR_INFO,
                             "Source stepping completed");
        return (0);
    }
}


/* Spice3 Source stepping
 * Modified 2005 - Paolo Nenzi (extracted from CKTop.c code)
 *
 * return value:
 * 0 -> method converged
 * 1 -> method failed
 *
 * Note that no path out of this code allows ckt->CKTsrcFact to be
 * anything but 1.00000.
 */
static int
spice3_src (CKTcircuit * ckt, long int firstmode,
            long int continuemode, int iterlim)
{

    int converged, i;

    NG_IGNORE(iterlim);

    ckt->CKTmode = firstmode;
    SPfrontEnd->IFerrorf (ERR_INFO,
                         "Starting source stepping");

    for (i = 0; i <= ckt->CKTnumSrcSteps; i++) {
        ckt->CKTsrcFact = ((double) i) / ((double) ckt->CKTnumSrcSteps);
#ifdef XSPICE
        /* gtri - begin - wbk - add convergence problem reporting flags */
        ckt->enh->conv_debug.last_NIiter_call = MIF_TRUE;
        /* gtri - end - wbk - add convergence problem reporting flags */
#endif
        converged = NIiter (ckt, ckt->CKTdcTrcvMaxIter);
        ckt->CKTmode = continuemode;
        if (converged != 0) {
            ckt->CKTsrcFact = 1;
            ckt->CKTcurrentAnalysis = DOING_TRAN;
            SPfrontEnd->IFerrorf (ERR_WARNING,
                                 "source stepping failed");
#ifdef XSPICE
            /* gtri - begin - wbk - add convergence problem reporting flags */
            ckt->enh->conv_debug.last_NIiter_call = MIF_FALSE;
            /* gtri - end - wbk - add convergence problem reporting flags */
#endif
            return (converged);
        }
        SPfrontEnd->IFerrorf (ERR_INFO,
                             "One successful source step");
    }
    SPfrontEnd->IFerrorf (ERR_INFO,
                         "Source stepping completed");
    ckt->CKTsrcFact = 1;
#ifdef XSPICE
    /* gtri - begin - wbk - add convergence problem reporting flags */
    ckt->enh->conv_debug.last_NIiter_call = MIF_FALSE;
    /* gtri - end - wbk - add convergence problem reporting flags */
#endif
    return (0);
}
