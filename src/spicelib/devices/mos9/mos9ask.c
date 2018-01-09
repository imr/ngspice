/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1987 Mathew Lew and Thomas L. Quarles
Modified: Alan Gillespie
**********/

#include "ngspice/ngspice.h"
#include "ngspice/const.h"
#include "ngspice/ifsim.h"
#include "ngspice/cktdefs.h"
#include "ngspice/devdefs.h"
#include "mos9defs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

/*ARGSUSED*/
int
MOS9ask(CKTcircuit *ckt, GENinstance *inst, int which, 
        IFvalue *value, IFvalue *select)
{
    MOS9instance *here = (MOS9instance *)inst;
    double vr;
    double vi;
    double sr;
    double si;
    double vm;
    static char *msg = "Current and power not available for ac analysis";
    switch(which) {
        case MOS9_TEMP:
            value->rValue = here->MOS9temp-CONSTCtoK;
            return(OK);
        case MOS9_CGS:
            value->rValue = 2* *(ckt->CKTstate0 + here->MOS9capgs);
            return(OK);
        case MOS9_CGD:
            value->rValue = 2* *(ckt->CKTstate0 + here->MOS9capgd);
            return(OK);
        case MOS9_M:
            value->rValue = here->MOS9m;
                return(OK);
        case MOS9_L:
            value->rValue = here->MOS9l;
                return(OK);
        case MOS9_W:
            value->rValue = here->MOS9w;
                return(OK);
        case MOS9_AS:
            value->rValue = here->MOS9sourceArea;
                return(OK);
        case MOS9_AD:
            value->rValue = here->MOS9drainArea;
                return(OK);
        case MOS9_PS:
            value->rValue = here->MOS9sourcePerimiter;
                return(OK);
        case MOS9_PD:
            value->rValue = here->MOS9drainPerimiter;
                return(OK);
        case MOS9_NRS:
            value->rValue = here->MOS9sourceSquares;
                return(OK);
        case MOS9_NRD:
            value->rValue = here->MOS9drainSquares;
                return(OK);
        case MOS9_OFF:
            value->rValue = here->MOS9off;
                return(OK);
        case MOS9_IC_VBS:
            value->rValue = here->MOS9icVBS;
                return(OK);
        case MOS9_IC_VDS:
            value->rValue = here->MOS9icVDS;
                return(OK);
        case MOS9_IC_VGS:
            value->rValue = here->MOS9icVGS;
            return(OK);
        case MOS9_DNODE:
            value->iValue = here->MOS9dNode;
            return(OK);
        case MOS9_GNODE:
            value->iValue = here->MOS9gNode;
            return(OK);
        case MOS9_SNODE:
            value->iValue = here->MOS9sNode;
            return(OK);
        case MOS9_BNODE:
            value->iValue = here->MOS9bNode;
            return(OK);
        case MOS9_DNODEPRIME:
            value->iValue = here->MOS9dNodePrime;
            return(OK);
        case MOS9_SNODEPRIME:
            value->iValue = here->MOS9sNodePrime;
            return(OK);
        case MOS9_SOURCECONDUCT:
	    value->rValue = here->MOS9sourceConductance;
            return(OK);
        case MOS9_DRAINCONDUCT:
	    value->rValue = here->MOS9drainConductance;
            return(OK);
        case MOS9_SOURCERESIST:
            if (here->MOS9sNodePrime != here->MOS9sNode)
		value->rValue = 1.0 / here->MOS9sourceConductance;
	    else
		value->rValue = 0.0;
            return(OK);
        case MOS9_DRAINRESIST:
            if (here->MOS9dNodePrime != here->MOS9dNode)
		value->rValue = 1.0 / here->MOS9drainConductance;
	    else
		value->rValue = 0.0;
            return(OK);
        case MOS9_VON:
            value->rValue = here->MOS9von;
            return(OK);
        case MOS9_VDSAT:
            value->rValue = here->MOS9vdsat;
            return(OK);
        case MOS9_SOURCEVCRIT:
            value->rValue = here->MOS9sourceVcrit;
            return(OK);
        case MOS9_DRAINVCRIT:
            value->rValue = here->MOS9drainVcrit;
            return(OK);
        case MOS9_CD:
            value->rValue = here->MOS9cd;
            return(OK);
        case MOS9_CBS:
            value->rValue = here->MOS9cbs;
            return(OK);
        case MOS9_CBD:
            value->rValue = here->MOS9cbd;
            return(OK);
        case MOS9_GMBS:
            value->rValue = here->MOS9gmbs;
            return(OK);
        case MOS9_GM:
            value->rValue = here->MOS9gm;
            return(OK);
        case MOS9_GDS:
            value->rValue = here->MOS9gds;
            return(OK);
        case MOS9_GBD:
            value->rValue = here->MOS9gbd;
            return(OK);
        case MOS9_GBS:
            value->rValue = here->MOS9gbs;
            return(OK);
        case MOS9_CAPBD:
            value->rValue = here->MOS9capbd;
            return(OK);
        case MOS9_CAPBS:
            value->rValue = here->MOS9capbs;
            return(OK);
        case MOS9_CAPZEROBIASBD:
            value->rValue = here->MOS9Cbd;
            return(OK);
        case MOS9_CAPZEROBIASBDSW:
            value->rValue = here->MOS9Cbdsw;
            return(OK);
        case MOS9_CAPZEROBIASBS:
            value->rValue = here->MOS9Cbs;
            return(OK);
        case MOS9_CAPZEROBIASBSSW:
            value->rValue = here->MOS9Cbssw;
            return(OK);
        case MOS9_VBD:
            value->rValue = *(ckt->CKTstate0 + here->MOS9vbd);
            return(OK);
        case MOS9_VBS:
            value->rValue = *(ckt->CKTstate0 + here->MOS9vbs);
            return(OK);
        case MOS9_VGS:
            value->rValue = *(ckt->CKTstate0 + here->MOS9vgs);
            return(OK);
        case MOS9_VDS:
            value->rValue = *(ckt->CKTstate0 + here->MOS9vds);
            return(OK);
        case MOS9_CAPGS:
            value->rValue = 2* *(ckt->CKTstate0 + here->MOS9capgs);
/* add overlap capacitance */
            value->rValue += (MOS9modPtr(here)->MOS9gateSourceOverlapCapFactor)
                             * here->MOS9m
                             * (here->MOS9w
                                +MOS9modPtr(here)->MOS9widthAdjust
                                -2*(MOS9modPtr(here)->MOS9widthNarrow));
            return(OK);
        case MOS9_QGS:
            value->rValue = *(ckt->CKTstate0 + here->MOS9qgs);
            return(OK);
        case MOS9_CQGS:
            value->rValue = *(ckt->CKTstate0 + here->MOS9cqgs);
            return(OK);
        case MOS9_CAPGD:
            value->rValue = 2* *(ckt->CKTstate0 + here->MOS9capgd);
/* add overlap capacitance */
            value->rValue += (MOS9modPtr(here)->MOS9gateDrainOverlapCapFactor)
                             * here->MOS9m
                             * (here->MOS9w
                                +MOS9modPtr(here)->MOS9widthAdjust
                                -2*(MOS9modPtr(here)->MOS9widthNarrow));
            return(OK);
        case MOS9_QGD:
            value->rValue = *(ckt->CKTstate0 + here->MOS9qgd);
            return(OK);
        case MOS9_CQGD:
            value->rValue = *(ckt->CKTstate0 + here->MOS9cqgd);
            return(OK);
        case MOS9_CAPGB:
            value->rValue = 2* *(ckt->CKTstate0 + here->MOS9capgb);
/* add overlap capacitance */
            value->rValue += (MOS9modPtr(here)->MOS9gateBulkOverlapCapFactor)
                             * here->MOS9m
                             * (here->MOS9l
                                +MOS9modPtr(here)->MOS9lengthAdjust
                                -2*(MOS9modPtr(here)->MOS9latDiff));
            return(OK);
        case MOS9_QGB:
            value->rValue = *(ckt->CKTstate0 + here->MOS9qgb);
            return(OK);
        case MOS9_CQGB:
            value->rValue = *(ckt->CKTstate0 + here->MOS9cqgb);
            return(OK);
        case MOS9_QBD:
            value->rValue = *(ckt->CKTstate0 + here->MOS9qbd);
            return(OK);
        case MOS9_CQBD:
            value->rValue = *(ckt->CKTstate0 + here->MOS9cqbd);
            return(OK);
        case MOS9_QBS:
            value->rValue = *(ckt->CKTstate0 + here->MOS9qbs);
            return(OK);
        case MOS9_CQBS:
            value->rValue = *(ckt->CKTstate0 + here->MOS9cqbs);
            return(OK);
        case MOS9_L_SENS_DC:
            if(ckt->CKTsenInfo){
                value->rValue = *(ckt->CKTsenInfo->SEN_Sap[select->iValue + 1]+
                        here->MOS9senParmNo);
            }
            return(OK);
        case MOS9_L_SENS_REAL:
            if(ckt->CKTsenInfo){
            value->rValue = *(ckt->CKTsenInfo->SEN_RHS[select->iValue + 1]+
                        here->MOS9senParmNo);
            }
            return(OK);
        case MOS9_L_SENS_IMAG:
            if(ckt->CKTsenInfo){
                value->rValue = *(ckt->CKTsenInfo->SEN_iRHS[select->iValue + 1]+
                        here->MOS9senParmNo);
            }
            return(OK);
        case MOS9_L_SENS_MAG:
            if(ckt->CKTsenInfo){
                vr = *(ckt->CKTrhsOld + select->iValue + 1); 
                vi = *(ckt->CKTirhsOld + select->iValue + 1); 
                vm = sqrt(vr*vr + vi*vi);
                if(vm == 0){
                    value->rValue = 0;
                    return(OK);
                }
                sr = *(ckt->CKTsenInfo->SEN_RHS[select->iValue + 1]+
                        here->MOS9senParmNo);
                si = *(ckt->CKTsenInfo->SEN_iRHS[select->iValue + 1]+
                        here->MOS9senParmNo);
                value->rValue = (vr * sr + vi * si)/vm;
            }
            return(OK);
        case MOS9_L_SENS_PH:
            if(ckt->CKTsenInfo){
                vr = *(ckt->CKTrhsOld + select->iValue + 1); 
                vi = *(ckt->CKTirhsOld + select->iValue + 1); 
                vm = vr*vr + vi*vi;
                if(vm == 0){
                    value->rValue = 0;
                    return(OK);
                }
                sr = *(ckt->CKTsenInfo->SEN_RHS[select->iValue + 1]+
                        here->MOS9senParmNo);
                si = *(ckt->CKTsenInfo->SEN_iRHS[select->iValue + 1]+
                        here->MOS9senParmNo);
                value->rValue =  (vr * si - vi * sr)/vm;
            }
            return(OK);
        case MOS9_L_SENS_CPLX:
            if(ckt->CKTsenInfo){
                value->cValue.real= 
                        *(ckt->CKTsenInfo->SEN_RHS[select->iValue + 1]+
                        here->MOS9senParmNo);
                value->cValue.imag= 
                        *(ckt->CKTsenInfo->SEN_iRHS[select->iValue + 1]+
                        here->MOS9senParmNo);
            }
            return(OK);
        case MOS9_W_SENS_DC:
            if(ckt->CKTsenInfo){
                value->rValue = *(ckt->CKTsenInfo->SEN_Sap[select->iValue + 1]+
                        here->MOS9senParmNo + here->MOS9sens_l);
            }
            return(OK);
        case MOS9_W_SENS_REAL:
            if(ckt->CKTsenInfo){
                value->rValue = *(ckt->CKTsenInfo->SEN_RHS[select->iValue + 1]+
                        here->MOS9senParmNo + here->MOS9sens_l);
            }
            return(OK);
        case MOS9_W_SENS_IMAG:
            if(ckt->CKTsenInfo){
                value->rValue = *(ckt->CKTsenInfo->SEN_iRHS[select->iValue + 1]+
                        here->MOS9senParmNo + here->MOS9sens_l);
            }
            return(OK);
        case MOS9_W_SENS_MAG:
            if(ckt->CKTsenInfo){
                vr = *(ckt->CKTrhsOld + select->iValue + 1); 
                vi = *(ckt->CKTirhsOld + select->iValue + 1); 
                vm = sqrt(vr*vr + vi*vi);
                if(vm == 0){
                    value->rValue = 0;
                    return(OK);
                }
                sr = *(ckt->CKTsenInfo->SEN_RHS[select->iValue + 1]+
                        here->MOS9senParmNo + here->MOS9sens_l);
                si = *(ckt->CKTsenInfo->SEN_iRHS[select->iValue + 1]+
                        here->MOS9senParmNo + here->MOS9sens_l);
                value->rValue = (vr * sr + vi * si)/vm;
            }
            return(OK);
        case MOS9_W_SENS_PH:
            if(ckt->CKTsenInfo){
                vr = *(ckt->CKTrhsOld + select->iValue + 1); 
                vi = *(ckt->CKTirhsOld + select->iValue + 1); 
                vm = vr*vr + vi*vi;
                if(vm == 0){
                    value->rValue = 0;
                    return(OK);
                }
                sr = *(ckt->CKTsenInfo->SEN_RHS[select->iValue + 1]+
                        here->MOS9senParmNo + here->MOS9sens_l);
                si = *(ckt->CKTsenInfo->SEN_iRHS[select->iValue + 1]+
                        here->MOS9senParmNo + here->MOS9sens_l);
                value->rValue =  (vr * si - vi * sr)/vm;
            }
            return(OK);
        case MOS9_W_SENS_CPLX:
            if(ckt->CKTsenInfo){
                value->cValue.real= 
                        *(ckt->CKTsenInfo->SEN_RHS[select->iValue + 1]+
                        here->MOS9senParmNo + here->MOS9sens_l);
                value->cValue.imag= 
                        *(ckt->CKTsenInfo->SEN_iRHS[select->iValue + 1]+
                        here->MOS9senParmNo + here->MOS9sens_l);
            }
            return(OK);
        case MOS9_CB :
            if (ckt->CKTcurrentAnalysis & DOING_AC) {
                errMsg = TMALLOC(char, strlen(msg) + 1);
                errRtn = "MOS9ask.c";
                strcpy(errMsg,msg);
                return(E_ASKCURRENT);
            } else {
                value->rValue = here->MOS9cbd + here->MOS9cbs - *(ckt->CKTstate0
                        + here->MOS9cqgb);
            }
            return(OK);
        case MOS9_CG :
            if (ckt->CKTcurrentAnalysis & DOING_AC) {
                errMsg = TMALLOC(char, strlen(msg) + 1);
                errRtn = "MOS9ask.c";
                strcpy(errMsg,msg);
                return(E_ASKCURRENT);
            } else if (ckt->CKTcurrentAnalysis & (DOING_DCOP | DOING_TRCV)) {
                value->rValue = 0;
            } else if ((ckt->CKTcurrentAnalysis & DOING_TRAN) && 
                        (ckt->CKTmode & MODETRANOP)) {
                value->rValue = 0;
            } else {
                value->rValue = *(ckt->CKTstate0 + here->MOS9cqgb) +
                    *(ckt->CKTstate0 + here->MOS9cqgd) + *(ckt->CKTstate0 + 
                    here->MOS9cqgs);
            }
            return(OK);
        case MOS9_CS :
            if (ckt->CKTcurrentAnalysis & DOING_AC) {
                errMsg = TMALLOC(char, strlen(msg) + 1);
                errRtn = "MOS9ask.c";
                strcpy(errMsg,msg);
                return(E_ASKCURRENT);
            } else {
                value->rValue = -here->MOS9cd;
                value->rValue -= here->MOS9cbd + here->MOS9cbs -
                        *(ckt->CKTstate0 + here->MOS9cqgb);
                if ((ckt->CKTcurrentAnalysis & DOING_TRAN) && 
                        !(ckt->CKTmode & MODETRANOP)) {
                    value->rValue -= *(ckt->CKTstate0 + here->MOS9cqgb) + 
                            *(ckt->CKTstate0 + here->MOS9cqgd) +
                            *(ckt->CKTstate0 + here->MOS9cqgs);
                }
            }
            return(OK);
        case MOS9_POWER :
            if (ckt->CKTcurrentAnalysis & DOING_AC) {
                errMsg = TMALLOC(char, strlen(msg) + 1);
                errRtn = "MOS9ask.c";
                strcpy(errMsg,msg);
                return(E_ASKPOWER);
            } else {
                double temp;

                value->rValue = here->MOS9cd * 
                        *(ckt->CKTrhsOld + here->MOS9dNode);
                value->rValue += (here->MOS9cbd + here->MOS9cbs -
                        *(ckt->CKTstate0 + here->MOS9cqgb)) *
                        *(ckt->CKTrhsOld + here->MOS9bNode);
                if ((ckt->CKTcurrentAnalysis & DOING_TRAN) && 
                        !(ckt->CKTmode & MODETRANOP)) {
                    value->rValue += (*(ckt->CKTstate0 + here->MOS9cqgb) + 
                            *(ckt->CKTstate0 + here->MOS9cqgd) + 
                            *(ckt->CKTstate0 + here->MOS9cqgs)) *
                            *(ckt->CKTrhsOld + here->MOS9gNode);
                }
                temp = -here->MOS9cd;
                temp -= here->MOS9cbd + here->MOS9cbs ;
                if ((ckt->CKTcurrentAnalysis & DOING_TRAN) && 
                        !(ckt->CKTmode & MODETRANOP)) {
                    temp -= *(ckt->CKTstate0 + here->MOS9cqgb) + 
                            *(ckt->CKTstate0 + here->MOS9cqgd) + 
                            *(ckt->CKTstate0 + here->MOS9cqgs);
                }
                value->rValue += temp * *(ckt->CKTrhsOld + here->MOS9sNode);
            }
            return(OK);
        default:
            return(E_BADPARM);
    }
    /* NOTREACHED */
}

