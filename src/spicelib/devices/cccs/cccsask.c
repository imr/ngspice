/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1987
**********/

/*
 * This routine gives access to the internal device parameters
 * of Current Controlled Current Source
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/devdefs.h"
#include "ngspice/ifsim.h"
#include "cccsdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


/* ARGSUSED */
int
CCCSask(CKTcircuit *ckt, GENinstance *inst, int which, IFvalue *value, IFvalue *select)
{
    CCCSinstance *here = (CCCSinstance*)inst;
    double vr;
    double vi;
    double sr;
    double si;
    double vm;
    static char *msg = "Current and power not available for ac analysis";

    switch(which) {
        case CCCS_GAIN:
            value->rValue = here->CCCScoeff;
            return (OK);
        case CCCS_CONTROL:
            value->uValue = here->CCCScontName;
            return (OK);
        case CCCS_M:
            value->rValue = here->CCCSmValue;
            return (OK);
        case CCCS_POS_NODE:
            value->iValue = here->CCCSposNode;
            return (OK);
        case CCCS_NEG_NODE:
            value->iValue = here->CCCSnegNode;
            return (OK);
        case CCCS_CONT_BR:
            value->iValue = here->CCCScontBranch;
            return (OK);
        case CCCS_CURRENT :
            if (ckt->CKTcurrentAnalysis & DOING_AC) {
                errMsg = TMALLOC(char, strlen(msg) + 1);
                errRtn = "CCCSask";
                strcpy(errMsg,msg);
                return(E_ASKCURRENT);
            } else {
                value->rValue = *(ckt->CKTrhsOld + 
                        here->CCCScontBranch) * here->CCCScoeff;
            }
            return(OK);
        case CCCS_VOLTS :
	    value->rValue = (*(ckt->CKTrhsOld + here->CCCSposNode) - 
		*(ckt->CKTrhsOld + here->CCCSnegNode));
            return(OK);
        case CCCS_POWER :
            if (ckt->CKTcurrentAnalysis & DOING_AC) {
                errMsg = TMALLOC(char, strlen(msg) + 1);
                errRtn = "CCCSask";
                strcpy(errMsg,msg);
                return(E_ASKPOWER);
            } else {
                value->rValue = *(ckt->CKTrhsOld + 
                        here->CCCScontBranch) * here->CCCScoeff * 
                        (*(ckt->CKTrhsOld + here->CCCSposNode) - 
                        *(ckt->CKTrhsOld + here->CCCSnegNode));
            }
            return(OK);
        case CCCS_QUEST_SENS_DC:
            if(ckt->CKTsenInfo){
                value->rValue = *(ckt->CKTsenInfo->SEN_Sap[select->iValue + 1]+
                        here->CCCSsenParmNo);
            }
            return(OK);
        case CCCS_QUEST_SENS_REAL:
            if(ckt->CKTsenInfo){
                value->rValue = *(ckt->CKTsenInfo->SEN_RHS[select->iValue + 1]+
                        here->CCCSsenParmNo);
            }
            return(OK);
        case CCCS_QUEST_SENS_IMAG:
            if(ckt->CKTsenInfo){
                value->rValue = *(ckt->CKTsenInfo->SEN_iRHS[select->iValue + 1]+
                    here->CCCSsenParmNo);
            }
            return(OK);
        case CCCS_QUEST_SENS_MAG:
            if(ckt->CKTsenInfo){
                vr = *(ckt->CKTrhsOld + select->iValue + 1); 
                vi = *(ckt->CKTirhsOld + select->iValue + 1); 
                vm = sqrt(vr*vr + vi*vi);
                if(vm == 0){
                    value->rValue = 0;
                    return(OK);
                }
                sr = *(ckt->CKTsenInfo->SEN_RHS[select->iValue + 1]+
                        here->CCCSsenParmNo);
                si = *(ckt->CKTsenInfo->SEN_iRHS[select->iValue + 1]+
                        here->CCCSsenParmNo);
                value->rValue = (vr * sr + vi * si)/vm;
            }
            return(OK);
        case CCCS_QUEST_SENS_PH:
            if(ckt->CKTsenInfo){
                vr = *(ckt->CKTrhsOld + select->iValue + 1); 
                vi = *(ckt->CKTirhsOld + select->iValue + 1); 
                vm = vr*vr + vi*vi;
                if(vm == 0){
                    value->rValue = 0;
                    return(OK);
                }
                sr = *(ckt->CKTsenInfo->SEN_RHS[select->iValue + 1]+
                        here->CCCSsenParmNo);
                si = *(ckt->CKTsenInfo->SEN_iRHS[select->iValue + 1]+
                        here->CCCSsenParmNo);
                value->rValue =  (vr * si - vi * sr)/vm;
            }
            return(OK);
        case CCCS_QUEST_SENS_CPLX:
            if(ckt->CKTsenInfo){
                value->cValue.real= 
                        *(ckt->CKTsenInfo->SEN_RHS[select->iValue + 1]+
                        here->CCCSsenParmNo);
                value->cValue.imag= 
                        *(ckt->CKTsenInfo->SEN_iRHS[select->iValue + 1]+
                        here->CCCSsenParmNo);
            }
            return(OK);
        default:
            return (E_BADPARM);
    }
    /* NOTREACHED */
}
