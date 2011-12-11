/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1987 Thomas L. Quarles
**********/

/*
 * This routine gives access to the internal device parameters
 * of Voltage Controlled Voltage Source
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/devdefs.h"
#include "ngspice/ifsim.h"
#include "vcvsdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

/* ARGSUSED */
int
VCVSask(CKTcircuit *ckt, GENinstance *inst, int which, IFvalue *value, IFvalue *select)
{
    VCVSinstance *here = (VCVSinstance *)inst;
    double vr;
    double vi;
    double sr;
    double si;
    double vm;
    static char *msg = "Current and power not available for ac analysis";
    switch(which) {
        case VCVS_POS_NODE:
            value->iValue = here->VCVSposNode;
            return (OK);
        case VCVS_NEG_NODE:
            value->iValue = here->VCVSnegNode;
            return (OK);
        case VCVS_CONT_P_NODE:
            value->iValue = here->VCVScontPosNode;
            return (OK);
        case VCVS_CONT_N_NODE:
            value->iValue = here->VCVScontNegNode;
            return (OK);
        case VCVS_IC:
            value->rValue = here->VCVSinitCond;
            return (OK);
        case VCVS_GAIN:
            value->rValue = here->VCVScoeff;
            return (OK);
        case VCVS_CONT_V_OLD:
            value->rValue = *(ckt->CKTstate0 + here->VCVScontVOld);
            return (OK);
        case VCVS_BR:
            value->iValue = here->VCVSbranch;
            return (OK);
        case VCVS_QUEST_SENS_DC:
            if(ckt->CKTsenInfo){
                value->rValue = *(ckt->CKTsenInfo->SEN_Sap[select->iValue + 1]+
                        here->VCVSsenParmNo);
            }
            return(OK);
        case VCVS_QUEST_SENS_REAL:
            if(ckt->CKTsenInfo){
                value->rValue = *(ckt->CKTsenInfo->SEN_RHS[select->iValue + 1]+
                        here->VCVSsenParmNo);
            }
            return(OK);
        case VCVS_QUEST_SENS_IMAG:
            if(ckt->CKTsenInfo){
                value->rValue = *(ckt->CKTsenInfo->SEN_iRHS[select->iValue + 1]+
                        here->VCVSsenParmNo);
            }
            return(OK);
        case VCVS_QUEST_SENS_MAG:
            if(ckt->CKTsenInfo){
                vr = *(ckt->CKTrhsOld + select->iValue + 1); 
                vi = *(ckt->CKTirhsOld + select->iValue + 1); 
                vm = sqrt(vr*vr + vi*vi);
                if(vm == 0){
                    value->rValue = 0;
                    return(OK);
                }
                sr = *(ckt->CKTsenInfo->SEN_RHS[select->iValue + 1]+
                        here->VCVSsenParmNo);
                si = *(ckt->CKTsenInfo->SEN_iRHS[select->iValue + 1]+
                        here->VCVSsenParmNo);
                value->rValue = (vr * sr + vi * si)/vm;
            }
            return(OK);
        case VCVS_QUEST_SENS_PH:
            if(ckt->CKTsenInfo){
                vr = *(ckt->CKTrhsOld + select->iValue + 1); 
                vi = *(ckt->CKTirhsOld + select->iValue + 1); 
                vm = vr*vr + vi*vi;
                if(vm == 0){
                    value->rValue = 0;
                    return(OK);
                }
                sr = *(ckt->CKTsenInfo->SEN_RHS[select->iValue + 1]+
                        here->VCVSsenParmNo);
                si = *(ckt->CKTsenInfo->SEN_iRHS[select->iValue + 1]+
                        here->VCVSsenParmNo);

                value->rValue =  (vr * si - vi * sr)/vm;
            }

            return(OK);
        case VCVS_QUEST_SENS_CPLX:
            if(ckt->CKTsenInfo){
                value->cValue.real= 
                        *(ckt->CKTsenInfo->SEN_RHS[select->iValue + 1]+
                        here->VCVSsenParmNo);
                value->cValue.imag= 
                        *(ckt->CKTsenInfo->SEN_iRHS[select->iValue + 1]+
                        here->VCVSsenParmNo);
            }
            return(OK);
        case VCVS_CURRENT :
            if (ckt->CKTcurrentAnalysis & DOING_AC) {
                errMsg = TMALLOC(char, strlen(msg) + 1);
                errRtn = "VCVSask";
                strcpy(errMsg,msg);
                return(E_ASKCURRENT);
            } else {
                value->rValue = *(ckt->CKTrhsOld + here->VCVSbranch);
            }
            return(OK);
        case VCVS_VOLTS :
	    value->rValue = (*(ckt->CKTrhsOld + here->VCVSposNode) - 
		*(ckt->CKTrhsOld + here->VCVSnegNode));
            return(OK);
        case VCVS_POWER :
            if (ckt->CKTcurrentAnalysis & DOING_AC) {
                errMsg = TMALLOC(char, strlen(msg) + 1);
                errRtn = "VCVSask";
                strcpy(errMsg,msg);
                return(E_ASKPOWER);
            } else {
                value->rValue = *(ckt->CKTrhsOld + here->VCVSbranch) *
                        (*(ckt->CKTrhsOld + here->VCVSposNode) - 
                        *(ckt->CKTrhsOld + here->VCVSnegNode));
            }
            return(OK);
        default:
            return (E_BADPARM);
    }
    /* NOTREACHED */
}
