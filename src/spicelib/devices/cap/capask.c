/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: September 2003 Paolo Nenzi
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/devdefs.h"
#include "capdefs.h"
#include "ngspice/ifsim.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

/* ARGSUSED */
int
CAPask(CKTcircuit *ckt, GENinstance *inst, int which, IFvalue *value,
       IFvalue *select)
{
    CAPinstance *here = (CAPinstance *)inst;
    double vr;
    double vi;
    double sr;
    double si;
    double vm;
    static char *msg = "Current and power not available for ac analysis";

    switch(which) {
        case CAP_TEMP:
            value->rValue = here->CAPtemp - CONSTCtoK;
            return(OK);
        case CAP_DTEMP:
            value->rValue = here->CAPdtemp;
            return(OK);
        case CAP_CAP:
            value->rValue=here->CAPcapac;
            value->rValue *= here->CAPm;
            return(OK);
        case CAP_IC:
            value->rValue = here->CAPinitCond;
            return(OK);
        case CAP_WIDTH:
            value->rValue = here->CAPwidth;
            return(OK);
        case CAP_LENGTH:
            value->rValue = here->CAPlength;
            return(OK);
        case CAP_SCALE:
            value->rValue = here->CAPscale;
            return(OK);
	case CAP_BV_MAX:
	    value->rValue = here->CAPbv_max;
	    return(OK);
        case CAP_M:
            value->rValue = here->CAPm;
            return(OK);
        case CAP_TC1:
            value->rValue = here->CAPtc1;
            return(OK);
        case CAP_TC2:
            value->rValue = here->CAPtc2;
            return(OK);
        case CAP_CURRENT:
            if (ckt->CKTcurrentAnalysis & DOING_AC) {
                errMsg = TMALLOC(char, strlen(msg) + 1);
                errRtn = "CAPask";
                strcpy(errMsg,msg);
                return(E_ASKCURRENT);
            } else if (ckt->CKTcurrentAnalysis & (DOING_DCOP | DOING_TRCV)) {
                value->rValue = 0;
            } else if (ckt->CKTcurrentAnalysis & DOING_TRAN) {
                if (ckt->CKTmode & MODETRANOP) {
                    value->rValue = 0;
                } else {
                    value->rValue = *(ckt->CKTstate0 + here->CAPccap);
                }
            } else value->rValue = *(ckt->CKTstate0 + here->CAPccap);

            value->rValue *= here->CAPm;

            return(OK);
        case CAP_POWER:
            if (ckt->CKTcurrentAnalysis & DOING_AC) {
                errMsg = TMALLOC(char, strlen(msg) + 1);
                errRtn = "CAPask";
                strcpy(errMsg,msg);
                return(E_ASKPOWER);
            } else if (ckt->CKTcurrentAnalysis & (DOING_DCOP | DOING_TRCV)) {
                value->rValue = 0;
            } else if (ckt->CKTcurrentAnalysis & DOING_TRAN) {
                if (ckt->CKTmode & MODETRANOP) {
                    value->rValue = 0;
                } else {
                    value->rValue = *(ckt->CKTstate0 + here->CAPccap) *
                            (*(ckt->CKTrhsOld + here->CAPposNode) -
                            *(ckt->CKTrhsOld + here->CAPnegNode));
                }
            } else value->rValue = *(ckt->CKTstate0 + here->CAPccap) *
                            (*(ckt->CKTrhsOld + here->CAPposNode) -
                            *(ckt->CKTrhsOld + here->CAPnegNode));

            value->rValue *= here->CAPm;

            return(OK);
        case CAP_QUEST_SENS_DC:
            if(ckt->CKTsenInfo){
               value->rValue = *(ckt->CKTsenInfo->SEN_Sap[select->iValue + 1]+
                    here->CAPsenParmNo);
            }
            return(OK);
        case CAP_QUEST_SENS_REAL:
            if(ckt->CKTsenInfo){
               value->rValue = *(ckt->CKTsenInfo->SEN_RHS[select->iValue + 1]+
                    here->CAPsenParmNo);
            }
            return(OK);
        case CAP_QUEST_SENS_IMAG:
            if(ckt->CKTsenInfo){
               value->rValue = *(ckt->CKTsenInfo->SEN_iRHS[select->iValue + 1]+
                    here->CAPsenParmNo);
            }
            return(OK);
        case CAP_QUEST_SENS_MAG:
            if(ckt->CKTsenInfo){
               vr = *(ckt->CKTrhsOld + select->iValue + 1);
               vi = *(ckt->CKTirhsOld + select->iValue + 1);
               vm = sqrt(vr*vr + vi*vi);
               if(vm == 0){
                 value->rValue = 0;
                 return(OK);
               }
               sr = *(ckt->CKTsenInfo->SEN_RHS[select->iValue + 1]+
                    here->CAPsenParmNo);
               si = *(ckt->CKTsenInfo->SEN_iRHS[select->iValue + 1]+
                    here->CAPsenParmNo);
                   value->rValue = (vr * sr + vi * si)/vm;
            }
            return(OK);
        case CAP_QUEST_SENS_PH:
            if(ckt->CKTsenInfo){
               vr = *(ckt->CKTrhsOld + select->iValue + 1);
               vi = *(ckt->CKTirhsOld + select->iValue + 1);
               vm = vr*vr + vi*vi;
               if(vm == 0){
                 value->rValue = 0;
                 return(OK);
               }
               sr = *(ckt->CKTsenInfo->SEN_RHS[select->iValue + 1]+
                    here->CAPsenParmNo);
               si = *(ckt->CKTsenInfo->SEN_iRHS[select->iValue + 1]+
                    here->CAPsenParmNo);

                   value->rValue = (vr * si - vi * sr)/vm;
            }
            return(OK);

        case CAP_QUEST_SENS_CPLX:
            if(ckt->CKTsenInfo){
                value->cValue.real=
                        *(ckt->CKTsenInfo->SEN_RHS[select->iValue + 1]+
                        here->CAPsenParmNo);
                value->cValue.imag=
                        *(ckt->CKTsenInfo->SEN_iRHS[select->iValue + 1]+
                        here->CAPsenParmNo);
            }
            return(OK);

        default:
            return(E_BADPARM);
        }
}

