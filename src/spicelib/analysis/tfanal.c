/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1988 Thomas L. Quarles
**********/

/* subroutine to do DC Transfer Function analysis     */

#include "ngspice.h"
#include "cktdefs.h"
#include "ifsim.h"
#include "sperror.h"
#include "smpdefs.h"
#include "tfdefs.h"


/* ARGSUSED */
int
TFanal(CKTcircuit *ckt, int restart) 
                    
                    /* forced restart flag */
{
    int size;
    int insrc = 0, outsrc = 0;
    double outputs[3];
    IFvalue outdata;    /* structure for output data vector, will point to 
                         * outputs vector above */
    IFvalue refval;     /* structure for 'reference' value (not used here) */
    int error;
    int converged;
    int i;
    void *plotptr = NULL;   /* pointer to out plot */
    void *ptr = NULL;
    IFuid uids[3];
    int Itype;
    int Vtype;
    char *name;
#define tfuid (uids[0]) /* unique id for the transfer function output */
#define inuid (uids[1]) /* unique id for the transfer function input imp. */
#define outuid (uids[2]) /* unique id for the transfer function out. imp. */


    /* first, find the operating point */
    converged = CKTop(ckt,
            (ckt->CKTmode & MODEUIC) | MODEDCOP | MODEINITJCT,
            (ckt->CKTmode & MODEUIC) | MODEDCOP | MODEINITFLOAT,
            ckt->CKTdcMaxIter);

    Itype = CKTtypelook("Isource");
    Vtype = CKTtypelook("Vsource");
    if(Itype != -1) {
        error = CKTfndDev((void*)ckt,&Itype,&ptr,
                ((TFan*)ckt->CKTcurJob)->TFinSrc, (void *)NULL,(IFuid)NULL);
        if(error ==0) {
            ((TFan*)ckt->CKTcurJob)->TFinIsI = 1;
            ((TFan*)ckt->CKTcurJob)->TFinIsV = 0;
        } else {
            ptr = NULL;
        }
    }

    if( (Vtype != -1) && (ptr==NULL) ) {
        error = CKTfndDev((void*)ckt,&Vtype,&ptr,
                ((TFan*)ckt->CKTcurJob)->TFinSrc, (void *)NULL,
                (IFuid)NULL);
        ((TFan*)ckt->CKTcurJob)->TFinIsV = 1;
        ((TFan*)ckt->CKTcurJob)->TFinIsI = 0;
        if(error !=0) {
            (*(SPfrontEnd->IFerror))(ERR_WARNING,
                    "Transfer function source %s not in circuit",
                    &(((TFan*)ckt->CKTcurJob)->TFinSrc));
            ((TFan*)ckt->CKTcurJob)->TFinIsV = 0;
            return(E_NOTFOUND);
        }
    }

    size = SMPmatSize(ckt->CKTmatrix);
    for(i=0;i<=size;i++) {
        ckt->CKTrhs[i] = 0;
    }

    if(((TFan*)ckt->CKTcurJob)->TFinIsI) {
        ckt->CKTrhs[((GENinstance*)ptr)->GENnode1] -= 1;
        ckt->CKTrhs[((GENinstance*)ptr)->GENnode2] += 1;
    } else {
        insrc = CKTfndBranch(ckt,((TFan*)ckt->CKTcurJob)->TFinSrc);
        ckt->CKTrhs[insrc] += 1;
    }


    SMPsolve(ckt->CKTmatrix,ckt->CKTrhs,ckt->CKTrhsSpare);
    ckt->CKTrhs[0]=0;

    /* make a UID for the transfer function output */
    (*(SPfrontEnd->IFnewUid))(ckt,&tfuid,(IFuid)NULL,"Transfer_function",
            UID_OTHER,(void **)NULL);

    /* make a UID for the input impedance */
    (*(SPfrontEnd->IFnewUid))(ckt,&inuid,((TFan*)ckt->CKTcurJob)->TFinSrc,
            "Input_impedance", UID_OTHER,(void **)NULL);

    /* make a UID for the output impedance */
    if(((TFan*)ckt->CKTcurJob)->TFoutIsI) {
        (*(SPfrontEnd->IFnewUid))(ckt,&outuid,((TFan*)ckt->CKTcurJob)->TFoutSrc
                ,"Output_impedance", UID_OTHER,(void **)NULL);
    } else {
        name = (char *)
    MALLOC(sizeof(char)*(strlen(((TFan*)ckt->CKTcurJob)->TFoutName)+22));
        (void)sprintf(name,"output_impedance_at_%s",
                ((TFan*)ckt->CKTcurJob)->TFoutName);
        (*(SPfrontEnd->IFnewUid))(ckt,&outuid,(IFuid)NULL,
                name, UID_OTHER,(void **)NULL);
    }

    error = (*(SPfrontEnd->OUTpBeginPlot))(ckt,(void *)(ckt->CKTcurJob),
            ((TFan*)(ckt->CKTcurJob))->JOBname,(IFuid)NULL,(int)0,3,
            uids,IF_REAL,&plotptr);
    if(error) return(error);

    /*find transfer function */
    if(((TFan*)ckt->CKTcurJob)->TFoutIsV) {
        outputs[0] = ckt->CKTrhs[((TFan*)ckt->CKTcurJob)->TFoutPos->number] -
                ckt->CKTrhs[((TFan*)ckt->CKTcurJob)->TFoutNeg->number] ;
    } else {
        outsrc = CKTfndBranch(ckt,((TFan*)ckt->CKTcurJob)->TFoutSrc);
        outputs[0] = ckt->CKTrhs[outsrc];
    }

    /* now for input resistance */
    if(((TFan*)ckt->CKTcurJob)->TFinIsI) {
        outputs[1] = ckt->CKTrhs[((GENinstance*)ptr)->GENnode2] -
                ckt->CKTrhs[((GENinstance*)ptr)->GENnode1];
    } else {
        if(fabs(ckt->CKTrhs[insrc])<1e-20) {
            outputs[1]=1e20;
        } else {
            outputs[1] = -1/ckt->CKTrhs[insrc];
        }
    }

    if(((TFan*)ckt->CKTcurJob)->TFoutIsI && 
            (((TFan*)ckt->CKTcurJob)->TFoutSrc == 
            ((TFan*)ckt->CKTcurJob)->TFinSrc)) {
        outputs[2]=outputs[1];
        goto done;
        /* no need to compute output resistance when it is the same as 
           the input  */
    }
    /* now for output resistance */
    for(i=0;i<=size;i++) {
        ckt->CKTrhs[i] = 0;
    }
    if(((TFan*)ckt->CKTcurJob)->TFoutIsV) {
        ckt->CKTrhs[((TFan*)ckt->CKTcurJob)->TFoutPos->number] -= 1;
        ckt->CKTrhs[((TFan*)ckt->CKTcurJob)->TFoutNeg->number] += 1;
    } else {
        ckt->CKTrhs[outsrc] += 1;
    }
    SMPsolve(ckt->CKTmatrix,ckt->CKTrhs,ckt->CKTrhsSpare);
    ckt->CKTrhs[0]=0;
    if(((TFan*)ckt->CKTcurJob)->TFoutIsV) {
        outputs[2]= ckt->CKTrhs[((TFan*)ckt->CKTcurJob)->TFoutNeg->number] -
                ckt->CKTrhs[((TFan*)ckt->CKTcurJob)->TFoutPos->number];
    } else {
        outputs[2] = 1/MAX(1e-20,ckt->CKTrhs[outsrc]);
    }
done:
    outdata.v.numValue=3;
    outdata.v.vec.rVec=outputs;
    refval.rValue = 0;
    (*(SPfrontEnd->OUTpData))(plotptr,&refval,&outdata);
    (*(SPfrontEnd->OUTendPlot))(plotptr);
    return(OK);
}


