/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1987 Thomas L. Quarles
**********/
/*
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/devdefs.h"
#include "ngspice/ifsim.h"
#include "mesdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


/* ARGSUSED */
int
MESask(CKTcircuit *ckt, GENinstance *inst, int which, IFvalue *value, IFvalue *select)
{
    MESinstance *here = (MESinstance*)inst;
    static char *msg = "Current and power not available in ac analysis";

    NG_IGNORE(select);

    switch(which) {
        case MES_AREA:
            value->rValue = here->MESarea;
            value->rValue *= here->MESm;
            return (OK);
        case MES_IC_VDS:
            value->rValue = here->MESicVDS;
            return (OK);
        case MES_IC_VGS:
            value->rValue = here->MESicVGS;
            return (OK);
        case MES_OFF:
            value->iValue = here->MESoff;
            return (OK);
        case MES_DRAINNODE:
            value->iValue = here->MESdrainNode;
            return (OK);
        case MES_GATENODE:
            value->iValue = here->MESgateNode;
            return (OK);
        case MES_SOURCENODE:
            value->iValue = here->MESsourceNode;
            return (OK);
        case MES_DRAINPRIMENODE:
            value->iValue = here->MESdrainPrimeNode;
            return (OK);
        case MES_VGS:
            value->rValue = *(ckt->CKTstate0 + here->MESvgs);
            return (OK);
        case MES_VGD:
            value->rValue = *(ckt->CKTstate0 + here->MESvgd);
            return (OK);
        case MES_CG:
            value->rValue = *(ckt->CKTstate0 + here->MEScg);
            value->rValue *= here->MESm;
            return (OK);
        case MES_CD:
            value->rValue = *(ckt->CKTstate0 + here->MEScd);
            value->rValue *= here->MESm;
            return (OK);
        case MES_CGD:
            value->rValue = *(ckt->CKTstate0 + here->MEScgd);
            value->rValue *= here->MESm;
            return (OK);
        case MES_GM:
            value->rValue = *(ckt->CKTstate0 + here->MESgm);
            value->rValue *= here->MESm;
            return (OK);
        case MES_GDS:
            value->rValue = *(ckt->CKTstate0 + here->MESgds);
            value->rValue *= here->MESm;
            return (OK);
        case MES_GGS:
            value->rValue = *(ckt->CKTstate0 + here->MESggs);
            value->rValue *= here->MESm;
            return (OK);
        case MES_GGD:
            value->rValue = *(ckt->CKTstate0 + here->MESggd);
            value->rValue *= here->MESm;
            return (OK);
        case MES_QGS:
            value->rValue = *(ckt->CKTstate0 + here->MESqgs);
            value->rValue *= here->MESm;
            return (OK);
        case MES_CQGS:
            value->rValue = *(ckt->CKTstate0 + here->MEScqgs);
            value->rValue *= here->MESm;
            return (OK);
        case MES_QGD:
            value->rValue = *(ckt->CKTstate0 + here->MESqgd);
            value->rValue *= here->MESm;
            return (OK);
        case MES_CQGD:
            value->rValue = *(ckt->CKTstate0 + here->MEScqgd);
            value->rValue *= here->MESm;
            return (OK);
        case MES_CS :
             if (ckt->CKTcurrentAnalysis & DOING_AC) {
                 errMsg = TMALLOC(char, strlen(msg) + 1);
                 errRtn = "MESask";
                 strcpy(errMsg,msg);
                 return(E_ASKCURRENT);
             } else {
                 value->rValue = -*(ckt->CKTstate0 + here->MEScd);
                 value->rValue -= *(ckt->CKTstate0 + here->MEScg);
                 value->rValue *= here->MESm;
             }
             return(OK);
        case MES_POWER :
             if (ckt->CKTcurrentAnalysis & DOING_AC) {
                 errMsg = TMALLOC(char, strlen(msg) + 1);
                 errRtn = "MESask";
                 strcpy(errMsg,msg);
                 return(E_ASKPOWER);
             } else {
                 value->rValue = *(ckt->CKTstate0 + here->MEScd) *
                         *(ckt->CKTrhsOld + here->MESdrainNode);
                 value->rValue += *(ckt->CKTstate0 + here->MEScg) *
                         *(ckt->CKTrhsOld + here->MESgateNode);
                 value->rValue -= (*(ckt->CKTstate0+here->MEScd) +
                         *(ckt->CKTstate0 + here->MEScg)) *
                         *(ckt->CKTrhsOld + here->MESsourceNode);
                 value->rValue *= here->MESm;
             }
             return(OK);
        default:
            return (E_BADPARM);
    }
    /* NOTREACHED */
}
