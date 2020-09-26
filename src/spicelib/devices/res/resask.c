/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: Apr 2000 - Paolo Nenzi
**********/

#include "ngspice/ngspice.h"
#include "ngspice/const.h"
#include "resdefs.h"
#include "ngspice/ifsim.h"
#include "ngspice/cktdefs.h"
#include "ngspice/sperror.h"


/* TODO : there are "double" value compared with 0 (eg: vm == 0)
 *        Need to substitute this check with a suitable eps.
 *        PN 2003
 */

int
RESask(CKTcircuit *ckt, GENinstance *inst, int which, IFvalue *value,
       IFvalue *select)
{
    RESinstance *fast = (RESinstance *)inst;
    double vr;
    double vi;
    double sr;
    double si;
    double vm;
    static char *msg = "Current and power not available for ac analysis";

    switch(which) {
    case RES_TEMP:
        value->rValue = fast->REStemp - CONSTCtoK;
        return(OK);
    case RES_DTEMP:
        value->rValue = fast->RESdtemp;
        return(OK);
    case RES_CONDUCT:
        value->rValue = fast->RESconduct;
        return(OK);
    case RES_RESIST:
        value->rValue = fast->RESresist;
        return(OK);
    case RES_ACCONDUCT:
        value->rValue = fast->RESacConduct;
        return (OK);
    case RES_ACRESIST:
        value->rValue = fast->RESacResist;
        return(OK);
    case RES_LENGTH:
        value->rValue = fast->RESlength;
        return(OK);
    case RES_WIDTH:
        value->rValue = fast->RESwidth;
        return(OK);
    case RES_SCALE:
        value->rValue = fast->RESscale;
        return(OK);
    case RES_M:
        value->rValue = fast->RESm;
        return(OK);
    case RES_TC1:
        value->rValue = fast->REStc1;
        return(OK);
    case RES_TC2:
        value->rValue = fast->REStc2;
        return(OK);
    case RES_TCE:
        value->rValue = fast->REStce;
        return(OK);
    case RES_BV_MAX:
        value->rValue = fast->RESbv_max;
        return(OK);
    case RES_NOISY:
        value->iValue = fast->RESnoisy;
        return(OK);
    case RES_QUEST_SENS_DC:
        if (ckt->CKTsenInfo) {
            value->rValue = *(ckt->CKTsenInfo->SEN_Sap[select->iValue + 1] +
                              fast->RESsenParmNo);
        }
        return(OK);
    case RES_QUEST_SENS_REAL:
        if (ckt->CKTsenInfo) {
            value->rValue = *(ckt->CKTsenInfo->SEN_RHS[select->iValue + 1] +
                              fast->RESsenParmNo);
        }
        return(OK);
    case RES_QUEST_SENS_IMAG:
        if (ckt->CKTsenInfo) {
            value->rValue = *(ckt->CKTsenInfo->SEN_iRHS[select->iValue + 1] +
                              fast->RESsenParmNo);
        }
        return(OK);
    case RES_QUEST_SENS_MAG:
        if (ckt->CKTsenInfo) {
            vr = *(ckt->CKTrhsOld + select->iValue + 1);
            vi = *(ckt->CKTirhsOld + select->iValue + 1);
            vm = sqrt(vr*vr + vi*vi);
            if (vm == 0) {
                value->rValue = 0;
                return(OK);
            }
            sr = *(ckt->CKTsenInfo->SEN_RHS[select->iValue + 1] +
                   fast->RESsenParmNo);
            si = *(ckt->CKTsenInfo->SEN_iRHS[select->iValue + 1] +
                   fast->RESsenParmNo);
            value->rValue = (vr * sr + vi * si) / vm;
        }
        return(OK);
    case RES_QUEST_SENS_PH:
        if (ckt->CKTsenInfo) {
            vr = *(ckt->CKTrhsOld + select->iValue + 1);
            vi = *(ckt->CKTirhsOld + select->iValue + 1);
            vm = vr*vr + vi*vi;
            if (vm == 0) {
                value->rValue = 0;
                return(OK);
            }
            sr = *(ckt->CKTsenInfo->SEN_RHS[select->iValue + 1] +
                   fast->RESsenParmNo);
            si = *(ckt->CKTsenInfo->SEN_iRHS[select->iValue + 1] +
                   fast->RESsenParmNo);
            value->rValue = (vr * si - vi * sr) / vm;
        }
        return(OK);
    case RES_QUEST_SENS_CPLX:
        if (ckt->CKTsenInfo) {
            value->cValue.real=
                *(ckt->CKTsenInfo->SEN_RHS[select->iValue + 1] +
                  fast->RESsenParmNo);
            value->cValue.imag=
                *(ckt->CKTsenInfo->SEN_iRHS[select->iValue + 1] +
                  fast->RESsenParmNo);
        }
        return(OK);
    case RES_CURRENT:
        if (ckt->CKTcurrentAnalysis & DOING_AC) {
            errMsg = tprintf("%s: %s", inst->GENname, msg);
            errRtn = "RESask";
            return(E_ASKCURRENT);
        } else if (ckt->CKTrhsOld) {
            value->rValue = (*(ckt->CKTrhsOld + fast->RESposNode) -
                             *(ckt->CKTrhsOld + fast->RESnegNode));
            value->rValue *= fast->RESconduct;
            return(OK);
        } else {
            errMsg = tprintf("No current values available for %s", fast->RESname);
            errRtn = "RESask";
            return(E_ASKCURRENT);
        }
    case RES_POWER:
        if (ckt->CKTcurrentAnalysis & DOING_AC) {
            errMsg = tprintf("%s: %s", inst->GENname, msg);
            errRtn = "RESask";
            return(E_ASKPOWER);
        } else if (ckt->CKTrhsOld) {
            value->rValue = (*(ckt->CKTrhsOld + fast->RESposNode) -
                             *(ckt->CKTrhsOld + fast->RESnegNode)) *
                            (*(ckt->CKTrhsOld + fast->RESposNode) -
                             *(ckt->CKTrhsOld + fast->RESnegNode));
            value->rValue *= fast->RESconduct;
            return(OK);
        } else {
            errMsg = tprintf("No power values available for %s", fast->RESname);
            errRtn = "RESask";
            return(E_ASKCURRENT);
        }
        
    default:
        return(E_BADPARM);
    }
    /* NOTREACHED */
}
