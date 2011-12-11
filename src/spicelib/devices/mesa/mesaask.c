/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1987 Thomas L. Quarles
**********/
/*
Imported into MESA model: 2001 Paolo Nenzi
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/devdefs.h"
#include "ngspice/ifsim.h"
#include "mesadefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


/* ARGSUSED */
int
MESAask(CKTcircuit *ckt, GENinstance *inst, int which, IFvalue *value, IFvalue *select)
{
    MESAinstance *here = (MESAinstance*)inst;
    static char *msg = "Current and power not available in ac analysis";

    NG_IGNORE(select);

    switch(which) {
        case MESA_LENGTH:
            value->rValue = here->MESAlength;
            return (OK);
        case MESA_WIDTH:
            value->rValue = here->MESAwidth;
            value->rValue *= here->MESAm;
            return (OK); 
        case MESA_M:
            value->rValue = here->MESAm;
            return (OK); 
        case MESA_IC_VDS:
            value->rValue = here->MESAicVDS;
            return (OK);
        case MESA_IC_VGS:
            value->rValue = here->MESAicVGS;
            return (OK);
        case MESA_OFF:
            value->iValue = here->MESAoff;
            return (OK);
        case MESA_TD:
            value->rValue = here->MESAtd - CONSTCtoK;
            return (OK);    
        case MESA_TS:
            value->rValue = here->MESAts - CONSTCtoK;
            return (OK);
        case MESA_DTEMP:
            value->rValue = here->MESAdtemp;
            return (OK);
        case MESA_DRAINNODE:
            value->iValue = here->MESAdrainNode;
            return (OK);
        case MESA_GATENODE:
            value->iValue = here->MESAgateNode;
            return (OK);
        case MESA_SOURCENODE:
            value->iValue = here->MESAsourceNode;
            return (OK);
        case MESA_DRAINPRIMENODE:
            value->iValue = here->MESAdrainPrimeNode;
            return (OK);
        case MESA_SOURCEPRIMENODE:
            value->iValue = here->MESAsourcePrimeNode;
            return (OK);
        case MESA_GATEPRIMENODE:
            value->iValue = here->MESAgatePrimeNode;
            return (OK);       
        case MESA_VGS:
            value->rValue = *(ckt->CKTstate0 + here->MESAvgs);
            return (OK);
        case MESA_VGD:
            value->rValue = *(ckt->CKTstate0 + here->MESAvgd);
            return (OK);
        case MESA_CG:
            value->rValue = *(ckt->CKTstate0 + here->MESAcg);
            value->rValue *= here->MESAm;
            return (OK);
        case MESA_CD:
            value->rValue = *(ckt->CKTstate0 + here->MESAcd);
            value->rValue *= here->MESAm;
            return (OK);
        case MESA_CGD:
            value->rValue = *(ckt->CKTstate0 + here->MESAcgd);
            value->rValue *= here->MESAm;
            return (OK);
        case MESA_GM:
            value->rValue = *(ckt->CKTstate0 + here->MESAgm);
            value->rValue *= here->MESAm;
            return (OK);
        case MESA_GDS:
            value->rValue = *(ckt->CKTstate0 + here->MESAgds);
            value->rValue *= here->MESAm;
            return (OK);
        case MESA_GGS:
            value->rValue = *(ckt->CKTstate0 + here->MESAggs);
            value->rValue *= here->MESAm;
            return (OK);
        case MESA_GGD:
            value->rValue = *(ckt->CKTstate0 + here->MESAggd);
            value->rValue *= here->MESAm;
            return (OK);
        case MESA_QGS:
            value->rValue = *(ckt->CKTstate0 + here->MESAqgs);
            value->rValue *= here->MESAm;
            return (OK);
        case MESA_CQGS:
            value->rValue = *(ckt->CKTstate0 + here->MESAcqgs);
            value->rValue *= here->MESAm;
            return (OK);
        case MESA_QGD:
            value->rValue = *(ckt->CKTstate0 + here->MESAqgd);
            value->rValue *= here->MESAm;
            return (OK);
        case MESA_CQGD:
            value->rValue = *(ckt->CKTstate0 + here->MESAcqgd);
            value->rValue *= here->MESAm;
            return (OK);
        case MESA_CS :
             if (ckt->CKTcurrentAnalysis & DOING_AC) {
                 errMsg = TMALLOC(char, strlen(msg) + 1);
                 errRtn = "MESAask";
                 strcpy(errMsg,msg);
                 return(E_ASKCURRENT);
             } else {
                 value->rValue = -*(ckt->CKTstate0 + here->MESAcd);
                 value->rValue -= *(ckt->CKTstate0 + here->MESAcg);
                 value->rValue *= here->MESAm;
             }
             return(OK);
        case MESA_POWER :
             if (ckt->CKTcurrentAnalysis & DOING_AC) {
                 errMsg = TMALLOC(char, strlen(msg) + 1);
                 errRtn = "MESAask";
                 strcpy(errMsg,msg);
                 return(E_ASKPOWER);
             } else {
                 value->rValue = *(ckt->CKTstate0 + here->MESAcd) *
                         *(ckt->CKTrhsOld + here->MESAdrainNode);
                 value->rValue += *(ckt->CKTstate0 + here->MESAcg) *
                         *(ckt->CKTrhsOld + here->MESAgateNode);
                 value->rValue -= (*(ckt->CKTstate0+here->MESAcd) +
                         *(ckt->CKTstate0 + here->MESAcg)) *
                         *(ckt->CKTrhsOld + here->MESAsourceNode);
                 value->rValue *= here->MESAm;
             }
             return(OK);
        default:
            return (E_BADPARM);
    }
    /* NOTREACHED */
}
