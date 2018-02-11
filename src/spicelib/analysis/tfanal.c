/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1988 Thomas L. Quarles
**********/

/* subroutine to do DC Transfer Function analysis     */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/ifsim.h"
#include "ngspice/sperror.h"
#include "ngspice/smpdefs.h"
#include "ngspice/tfdefs.h"


/* ARGSUSED */
int
TFanal(CKTcircuit *ckt, int restart) 
                    
                    /* forced restart flag */
{
    TFan *job = (TFan *) ckt->CKTcurJob;

    int size;
    int insrc = 0, outsrc = 0;
    double outputs[3];
    IFvalue outdata;    /* structure for output data vector, will point to 
                         * outputs vector above */
    IFvalue refval;     /* structure for 'reference' value (not used here) */
    int error;
    int converged;
    int i;
    runDesc *plotptr = NULL;   /* pointer to out plot */
    GENinstance *ptr = NULL;
    IFuid uids[3];
    char *name;
#define tfuid (uids[0]) /* unique id for the transfer function output */
#define inuid (uids[1]) /* unique id for the transfer function input imp. */
#define outuid (uids[2]) /* unique id for the transfer function out. imp. */

    NG_IGNORE(restart);

    /* first, find the operating point */
    converged = CKTop(ckt,
            (ckt->CKTmode & MODEUIC) | MODEDCOP | MODEINITJCT,
            (ckt->CKTmode & MODEUIC) | MODEDCOP | MODEINITFLOAT,
            ckt->CKTdcMaxIter);

    ptr = CKTfndDev(ckt, job->TFinSrc);

    if (!ptr || ptr->GENmodPtr->GENmodType < 0) {
        SPfrontEnd->IFerrorf (ERR_WARNING,
                             "Transfer function source %s not in circuit",
                             job->TFinSrc);
        job->TFinIsV = 0;
        job->TFinIsI = 0;
        return E_NOTFOUND;
    }

    if (ptr->GENmodPtr->GENmodType == CKTtypelook("Vsource")) {
        job->TFinIsV = 1;
        job->TFinIsI = 0;
    } else if (ptr->GENmodPtr->GENmodType == CKTtypelook("Isource")) {
        job->TFinIsV = 0;
        job->TFinIsI = 1;
    } else {
        SPfrontEnd->IFerrorf (ERR_WARNING,
                             "Transfer function source %s not of proper type",
                             job->TFinSrc);
        return E_NOTFOUND;
    }

    size = SMPmatSize(ckt->CKTmatrix);
    for(i=0;i<=size;i++) {
        ckt->CKTrhs[i] = 0;
    }

    if (job->TFinIsI) {
        ckt->CKTrhs[GENnode(ptr)[0]] -= 1;
        ckt->CKTrhs[GENnode(ptr)[1]] += 1;
    } else {
        insrc = CKTfndBranch(ckt, job->TFinSrc);
        ckt->CKTrhs[insrc] += 1;
    }


    SMPsolve(ckt->CKTmatrix,ckt->CKTrhs,ckt->CKTrhsSpare);
    ckt->CKTrhs[0]=0;

    /* make a UID for the transfer function output */
    SPfrontEnd->IFnewUid (ckt, &tfuid, NULL, "Transfer_function", UID_OTHER, NULL);

    /* make a UID for the input impedance */
    SPfrontEnd->IFnewUid (ckt, &inuid, job->TFinSrc, "Input_impedance", UID_OTHER, NULL);

    /* make a UID for the output impedance */
    if (job->TFoutIsI) {
        SPfrontEnd->IFnewUid (ckt, &outuid, job->TFoutSrc ,"Output_impedance", UID_OTHER, NULL);
    } else {
        name = tprintf("output_impedance_at_%s", job->TFoutName);
        SPfrontEnd->IFnewUid (ckt, &outuid, NULL, name, UID_OTHER, NULL);
    }

    error = SPfrontEnd->OUTpBeginPlot (ckt, ckt->CKTcurJob,
                                       job->JOBname,
                                       NULL, 0,
                                       3, uids, IF_REAL,
                                       &plotptr);
    if(error) return(error);

    /*find transfer function */
    if (job->TFoutIsV) {
        outputs[0] = ckt->CKTrhs[job->TFoutPos->number] -
            ckt->CKTrhs[job->TFoutNeg->number];
    } else {
        outsrc = CKTfndBranch(ckt, job->TFoutSrc);
        outputs[0] = ckt->CKTrhs[outsrc];
    }

    /* now for input resistance */
    if (job->TFinIsI) {
        outputs[1] = ckt->CKTrhs[GENnode(ptr)[1]] -
            ckt->CKTrhs[GENnode(ptr)[0]];
    } else {
        if(fabs(ckt->CKTrhs[insrc])<1e-20) {
            outputs[1]=1e20;
        } else {
            outputs[1] = -1/ckt->CKTrhs[insrc];
        }
    }

    if (job->TFoutIsI &&
            (job->TFoutSrc ==
            job->TFinSrc)) {
        outputs[2]=outputs[1];
        goto done;
        /* no need to compute output resistance when it is the same as 
           the input  */
    }
    /* now for output resistance */
    for(i=0;i<=size;i++) {
        ckt->CKTrhs[i] = 0;
    }
    if (job->TFoutIsV) {
        ckt->CKTrhs[job->TFoutPos->number] -= 1;
        ckt->CKTrhs[job->TFoutNeg->number] += 1;
    } else {
        ckt->CKTrhs[outsrc] += 1;
    }
    SMPsolve(ckt->CKTmatrix,ckt->CKTrhs,ckt->CKTrhsSpare);
    ckt->CKTrhs[0]=0;
    if (job->TFoutIsV) {
        outputs[2] = ckt->CKTrhs[job->TFoutNeg->number] -
            ckt->CKTrhs[job->TFoutPos->number];
    } else {
        outputs[2] = 1/MAX(1e-20,ckt->CKTrhs[outsrc]);
    }
done:
    outdata.v.numValue=3;
    outdata.v.vec.rVec=outputs;
    refval.rValue = 0;
    SPfrontEnd->OUTpData (plotptr, &refval, &outdata);
    SPfrontEnd->OUTendPlot (plotptr);
    return(OK);
}


