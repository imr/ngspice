/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1987 Thomas L. Quarles
**********/
/*
Imported into HFETA source: Paolo Nenzi 2001
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/devdefs.h"
#include "ngspice/ifsim.h"
#include "hfetdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


/* ARGSUSED */
int
HFETAask(CKTcircuit *ckt, GENinstance *inst, int which, IFvalue *value, IFvalue *select)
{
    HFETAinstance *here = (HFETAinstance*)inst;
    static char *msg = "Current and power not available in ac analysis";

    NG_IGNORE(select);

    switch(which) {
        case HFETA_LENGTH:
            value->rValue = here->HFETAlength;
            return (OK);
        case HFETA_WIDTH:
            value->rValue = here->HFETAwidth;
            value->rValue *= here->HFETAm;
            return (OK);
        case HFETA_M:
            value->rValue = here->HFETAm;
            return (OK);
        case HFETA_IC_VDS:
            value->rValue = here->HFETAicVDS;
            return (OK);
        case HFETA_IC_VGS:
            value->rValue = here->HFETAicVGS;
            return (OK);
        case HFETA_OFF:
            value->iValue = here->HFETAoff;
            return (OK);
        case HFETA_DRAINNODE:
            value->iValue = here->HFETAdrainNode;
            return (OK);
        case HFETA_GATENODE:
            value->iValue = here->HFETAgateNode;
            return (OK);
        case HFETA_SOURCENODE:
            value->iValue = here->HFETAsourceNode;
            return (OK);
        case HFETA_DRAINPRIMENODE:
            value->iValue = here->HFETAdrainPrimeNode;
            return (OK);
        case HFETA_SOURCEPRIMENODE:
            value->iValue = here->HFETAsourcePrimeNode;
            return (OK); 
        case HFETA_TEMP:
            value->rValue = here->HFETAtemp - CONSTCtoK;
            return(OK); 
        case HFETA_DTEMP:
            value->rValue = here->HFETAdtemp;
            return(OK);
        case HFETA_VGS:
            value->rValue = *(ckt->CKTstate0 + here->HFETAvgs);
            return (OK);
        case HFETA_VGD:
            value->rValue = *(ckt->CKTstate0 + here->HFETAvgd);
            return (OK);
        case HFETA_CG:
            value->rValue = *(ckt->CKTstate0 + here->HFETAcg);
            value->rValue *= here->HFETAm;
            return (OK);
        case HFETA_CD:
            value->rValue = *(ckt->CKTstate0 + here->HFETAcd);
            value->rValue *= here->HFETAm;
            return (OK);
        case HFETA_CGD:
            value->rValue = *(ckt->CKTstate0 + here->HFETAcgd);
            value->rValue *= here->HFETAm;
            return (OK);
        case HFETA_GM:
            value->rValue = *(ckt->CKTstate0 + here->HFETAgm);
            value->rValue *= here->HFETAm;
            return (OK);
        case HFETA_GDS:
            value->rValue = *(ckt->CKTstate0 + here->HFETAgds);
            value->rValue *= here->HFETAm;
            return (OK);
        case HFETA_GGS:
            value->rValue = *(ckt->CKTstate0 + here->HFETAggs);
            value->rValue *= here->HFETAm;
            return (OK);
        case HFETA_GGD:
            value->rValue = *(ckt->CKTstate0 + here->HFETAggd);
            value->rValue *= here->HFETAm;
            return (OK);
        case HFETA_QGS:
            value->rValue = *(ckt->CKTstate0 + here->HFETAqgs);
            value->rValue *= here->HFETAm;
            return (OK);
        case HFETA_CQGS:
            value->rValue = *(ckt->CKTstate0 + here->HFETAcqgs);
            value->rValue *= here->HFETAm;
            return (OK);
        case HFETA_QGD:
            value->rValue = *(ckt->CKTstate0 + here->HFETAqgd);
            value->rValue *= here->HFETAm;
            return (OK);
        case HFETA_CQGD:
            value->rValue = *(ckt->CKTstate0 + here->HFETAcqgd);
            value->rValue *= here->HFETAm;
            return (OK);
        case HFETA_CS :
             if (ckt->CKTcurrentAnalysis & DOING_AC) {
                 errMsg = TMALLOC(char, strlen(msg) + 1);
                 errRtn = "HFETAask";
                 strcpy(errMsg,msg);
                 return(E_ASKCURRENT);
             } else {
                 value->rValue = -*(ckt->CKTstate0 + here->HFETAcd);
                 value->rValue -= *(ckt->CKTstate0 + here->HFETAcg);
                 value->rValue *= here->HFETAm;
             }
             return(OK);
        case HFETA_POWER :
             if (ckt->CKTcurrentAnalysis & DOING_AC) {
                 errMsg = TMALLOC(char, strlen(msg) + 1);
                 errRtn = "HFETAask";
                 strcpy(errMsg,msg);
                 return(E_ASKPOWER);
             } else {
                 value->rValue = *(ckt->CKTstate0 + here->HFETAcd) *
                         *(ckt->CKTrhsOld + here->HFETAdrainNode);
                 value->rValue += *(ckt->CKTstate0 + here->HFETAcg) *
                         *(ckt->CKTrhsOld + here->HFETAgateNode);
                 value->rValue -= (*(ckt->CKTstate0+here->HFETAcd) +
                         *(ckt->CKTstate0 + here->HFETAcg)) *
                         *(ckt->CKTrhsOld + here->HFETAsourceNode);
                 value->rValue *= here->HFETAm;
             }
             return(OK);
        default:
            return (E_BADPARM);
    }
    /* NOTREACHED */
}
