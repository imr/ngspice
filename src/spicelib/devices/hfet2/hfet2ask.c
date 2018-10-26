/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1987 Thomas L. Quarles
**********/
/*
Imported into HFET2 source: Paolo Nenzi 2001
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/devdefs.h"
#include "ngspice/ifsim.h"
#include "hfet2defs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


/* ARGSUSED */
int
HFET2ask(CKTcircuit *ckt, GENinstance *inst, int which, IFvalue *value, 
         IFvalue *select)
{
    HFET2instance *here = (HFET2instance*)inst;
    static char *msg = "Current and power not available in ac analysis";

    NG_IGNORE(select);

    switch(which) {
        case HFET2_LENGTH:
            value->rValue = here->HFET2length;
            return (OK);
        case HFET2_WIDTH:
            value->rValue = here->HFET2width;
            value->rValue *= here->HFET2m;
            return (OK);
        case HFET2_IC_VDS:
            value->rValue = here->HFET2icVDS;
            return (OK);
        case HFET2_IC_VGS:
            value->rValue = here->HFET2icVGS;
            return (OK);
        case HFET2_OFF:
            value->iValue = here->HFET2off;
            return (OK);
        case HFET2_DRAINNODE:
            value->iValue = here->HFET2drainNode;
            return (OK);
        case HFET2_GATENODE:
            value->iValue = here->HFET2gateNode;
            return (OK);
        case HFET2_SOURCENODE:
            value->iValue = here->HFET2sourceNode;
            return (OK);
        case HFET2_DRAINPRIMENODE:
            value->iValue = here->HFET2drainPrimeNode;
            return (OK);
        case HFET2_SOURCEPRIMENODE:
            value->iValue = here->HFET2sourcePrimeNode;
            return (OK); 
        case HFET2_TEMP:
            value->rValue = here->HFET2temp - CONSTCtoK;
            return (OK); 
        case HFET2_DTEMP:
            value->rValue = here->HFET2dtemp;
            return (OK); 
        case HFET2_VGS:
            value->rValue = *(ckt->CKTstate0 + here->HFET2vgs);
            return (OK);
        case HFET2_VGD:
            value->rValue = *(ckt->CKTstate0 + here->HFET2vgd);
            return (OK);
        case HFET2_CG:
            value->rValue = *(ckt->CKTstate0 + here->HFET2cg);
            value->rValue *= here->HFET2m;
            return (OK);
        case HFET2_CD:
            value->rValue = *(ckt->CKTstate0 + here->HFET2cd);
            value->rValue *= here->HFET2m;
            return (OK);
        case HFET2_CGD:
            value->rValue = *(ckt->CKTstate0 + here->HFET2cgd);
            value->rValue *= here->HFET2m;
            return (OK);
        case HFET2_GM:
            value->rValue = *(ckt->CKTstate0 + here->HFET2gm);
            value->rValue *= here->HFET2m;
            return (OK);
        case HFET2_GDS:
            value->rValue = *(ckt->CKTstate0 + here->HFET2gds);
            value->rValue *= here->HFET2m;
            return (OK);
        case HFET2_GGS:
            value->rValue = *(ckt->CKTstate0 + here->HFET2ggs);
            value->rValue *= here->HFET2m;
            return (OK);
        case HFET2_GGD:
            value->rValue = *(ckt->CKTstate0 + here->HFET2ggd);
            value->rValue *= here->HFET2m;
            return (OK);
        case HFET2_QGS:
            value->rValue = *(ckt->CKTstate0 + here->HFET2qgs);
            value->rValue *= here->HFET2m;
            return (OK);
        case HFET2_CQGS:
            value->rValue = *(ckt->CKTstate0 + here->HFET2cqgs);
            value->rValue *= here->HFET2m;
            return (OK);
        case HFET2_QGD:
            value->rValue = *(ckt->CKTstate0 + here->HFET2qgd);
            value->rValue *= here->HFET2m;
            return (OK);
        case HFET2_CQGD:
            value->rValue = *(ckt->CKTstate0 + here->HFET2cqgd);
            value->rValue *= here->HFET2m;
            return (OK);
        case HFET2_CS :
             if (ckt->CKTcurrentAnalysis & DOING_AC) {
                 errMsg = TMALLOC(char, strlen(msg) + 1);
                 errRtn = "HFET2ask";
                 strcpy(errMsg,msg);
                 return(E_ASKCURRENT);
             } else {
                 value->rValue = -*(ckt->CKTstate0 + here->HFET2cd);
                 value->rValue -= *(ckt->CKTstate0 + here->HFET2cg);
                 value->rValue *= here->HFET2m;
             }
             return(OK);
        case HFET2_POWER :
             if (ckt->CKTcurrentAnalysis & DOING_AC) {
                 errMsg = TMALLOC(char, strlen(msg) + 1);
                 errRtn = "HFET2ask";
                 strcpy(errMsg,msg);
                 return(E_ASKPOWER);
             } else {
                 value->rValue = *(ckt->CKTstate0 + here->HFET2cd) *
                         *(ckt->CKTrhsOld + here->HFET2drainNode);
                 value->rValue += *(ckt->CKTstate0 + here->HFET2cg) *
                         *(ckt->CKTrhsOld + here->HFET2gateNode);
                 value->rValue -= (*(ckt->CKTstate0+here->HFET2cd) +
                         *(ckt->CKTstate0 + here->HFET2cg)) *
                         *(ckt->CKTrhsOld + here->HFET2sourceNode);
                 value->rValue *= here->HFET2m;
             }
             return(OK);
        default:
            return (E_BADPARM);
    }
    /* NOTREACHED */
}
