/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1987 Mathew Lew and Thomas L. Quarles
Modified: 2000 AlansFixes
**********/

#include "ngspice/ngspice.h"
#include "ngspice/const.h"
#include "ngspice/ifsim.h"
#include "ngspice/cktdefs.h"
#include "ngspice/devdefs.h"
#include "mos2defs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

/*ARGSUSED*/
int
MOS2ask(CKTcircuit *ckt, GENinstance *inst, int which, IFvalue *value,
        IFvalue *select)
{
    MOS2instance *here = (MOS2instance *)inst;
    double vr;
    double vi;
    double sr;
    double si;
    double vm;
    static char *msg = "Current and power not available for ac analysis";
    switch(which) {
        case MOS2_TEMP:
            value->rValue = here->MOS2temp-CONSTCtoK;
            return(OK);
        case MOS2_DTEMP:
            value->rValue = here->MOS2dtemp;
            return(OK);
        case MOS2_CGS:
            value->rValue = 2* *(ckt->CKTstate0 + here->MOS2capgs);
            return(OK);
        case MOS2_CGD:
            value->rValue = 2* *(ckt->CKTstate0 + here->MOS2capgd);
            return(OK);
        case MOS2_M:
            value->rValue = here->MOS2m;
            return(OK);
        case MOS2_L:
            value->rValue = here->MOS2l;
            return(OK);
        case MOS2_W:
            value->rValue = here->MOS2w;
                return(OK);
        case MOS2_AS:
            value->rValue = here->MOS2sourceArea;
                return(OK);
        case MOS2_AD:
            value->rValue = here->MOS2drainArea;
                return(OK);
        case MOS2_PS:
            value->rValue = here->MOS2sourcePerimiter;
                return(OK);
        case MOS2_PD:
            value->rValue = here->MOS2drainPerimiter;
                return(OK);
        case MOS2_NRS:
            value->rValue = here->MOS2sourceSquares;
                return(OK);
        case MOS2_NRD:
            value->rValue = here->MOS2drainSquares;
                return(OK);
        case MOS2_OFF:
            value->rValue = here->MOS2off;
                return(OK);
        case MOS2_IC_VBS:
            value->rValue = here->MOS2icVBS;
                return(OK);
        case MOS2_IC_VDS:
            value->rValue = here->MOS2icVDS;
                return(OK);
        case MOS2_IC_VGS:
            value->rValue = here->MOS2icVGS;
            return(OK);
        case MOS2_DNODE:
            value->iValue = here->MOS2dNode;
            return(OK);
        case MOS2_GNODE:
            value->iValue = here->MOS2gNode;
            return(OK);
        case MOS2_SNODE:
            value->iValue = here->MOS2sNode;
            return(OK);
        case MOS2_BNODE:
            value->iValue = here->MOS2bNode;
            return(OK);
        case MOS2_DNODEPRIME:
            value->iValue = here->MOS2dNodePrime;
            return(OK);
        case MOS2_SNODEPRIME:
            value->iValue = here->MOS2sNodePrime;
            return(OK);
        case MOS2_SOURCECONDUCT:
            value->rValue = here->MOS2sourceConductance;
            return(OK);
        case MOS2_DRAINCONDUCT:
            value->rValue = here->MOS2drainConductance;
            return(OK);
        case MOS2_SOURCERESIST:
	    if (here->MOS2sNodePrime != here->MOS2sNode)
		value->rValue = 1.0 / here->MOS2sourceConductance;
	    else
		value->rValue = 0.0;
            return(OK);
        case MOS2_DRAINRESIST:
	    if (here->MOS2dNodePrime != here->MOS2dNode)
		value->rValue = 1.0 / here->MOS2drainConductance;
	    else
		value->rValue = 0.0;
            return(OK);
        case MOS2_VON:
            value->rValue = here->MOS2von;
            return(OK);
        case MOS2_VDSAT:
            value->rValue = here->MOS2vdsat;
            return(OK);
        case MOS2_SOURCEVCRIT:
            value->rValue = here->MOS2sourceVcrit;
            return(OK);
        case MOS2_DRAINVCRIT:
            value->rValue = here->MOS2drainVcrit;
            return(OK);
        case MOS2_CD:
            value->rValue = here->MOS2cd;
            return(OK);
        case MOS2_CBS:
            value->rValue = here->MOS2cbs;
            return(OK);
        case MOS2_CBD:
            value->rValue = here->MOS2cbd;
            return(OK);
        case MOS2_GMBS:
            value->rValue = here->MOS2gmbs;
            return(OK);
        case MOS2_GM:
            value->rValue = here->MOS2gm;
            return(OK);
        case MOS2_GDS:
            value->rValue = here->MOS2gds;
            return(OK);
        case MOS2_GBD:
            value->rValue = here->MOS2gbd;
            return(OK);
        case MOS2_GBS:
            value->rValue = here->MOS2gbs;
            return(OK);
        case MOS2_CAPBD:
            value->rValue = here->MOS2capbd;
            return(OK);
        case MOS2_CAPBS:
            value->rValue = here->MOS2capbs;
            return(OK);
        case MOS2_CAPZEROBIASBD:
            value->rValue = here->MOS2Cbd;
            return(OK);
        case MOS2_CAPZEROBIASBDSW:
            value->rValue = here->MOS2Cbdsw;
            return(OK);
        case MOS2_CAPZEROBIASBS:
            value->rValue = here->MOS2Cbs;
            return(OK);
        case MOS2_CAPZEROBIASBSSW:
            value->rValue = here->MOS2Cbssw;
            return(OK);
        case MOS2_VBD:
            value->rValue = *(ckt->CKTstate0 + here->MOS2vbd);
            return(OK);
        case MOS2_VBS:
            value->rValue = *(ckt->CKTstate0 + here->MOS2vbs);
            return(OK);
        case MOS2_VGS:
            value->rValue = *(ckt->CKTstate0 + here->MOS2vgs);
            return(OK);
        case MOS2_VDS:
            value->rValue = *(ckt->CKTstate0 + here->MOS2vds);
            return(OK);
        case MOS2_CAPGS:
             value->rValue = 2* *(ckt->CKTstate0 + here->MOS2capgs);
/* add overlap capacitance */
            value->rValue += (MOS2modPtr(here)->MOS2gateSourceOverlapCapFactor)
                             * here->MOS2m
                             * (here->MOS2w);
            return(OK);
        case MOS2_QGS:
            value->rValue = *(ckt->CKTstate0 + here->MOS2qgs);
            return(OK);
        case MOS2_CQGS:
            value->rValue = *(ckt->CKTstate0 + here->MOS2cqgs);
            return(OK);
        case MOS2_CAPGD:
            value->rValue = 2* *(ckt->CKTstate0 + here->MOS2capgd);
/* add overlap capacitance */
            value->rValue += (MOS2modPtr(here)->MOS2gateDrainOverlapCapFactor)
                             * here->MOS2m
                             * (here->MOS2w);
            return(OK);
        case MOS2_QGD:
            value->rValue = *(ckt->CKTstate0 + here->MOS2qgd);
            return(OK);
        case MOS2_CQGD:
            value->rValue = *(ckt->CKTstate0 + here->MOS2cqgd);
            return(OK);
        case MOS2_CAPGB:
            value->rValue = 2* *(ckt->CKTstate0 + here->MOS2capgb);
/* add overlap capacitance */
            value->rValue += (MOS2modPtr(here)->MOS2gateBulkOverlapCapFactor)
                             * here->MOS2m
                             * (here->MOS2l
                                -2*(MOS2modPtr(here)->MOS2latDiff));
            return(OK);
        case MOS2_QGB:
            value->rValue = *(ckt->CKTstate0 + here->MOS2qgb);
            return(OK);
        case MOS2_CQGB:
            value->rValue = *(ckt->CKTstate0 + here->MOS2cqgb);
            return(OK);
        case MOS2_QBD:
            value->rValue = *(ckt->CKTstate0 + here->MOS2qbd);
            return(OK);
        case MOS2_CQBD:
            value->rValue = *(ckt->CKTstate0 + here->MOS2cqbd);
            return(OK);
        case MOS2_QBS:
            value->rValue = *(ckt->CKTstate0 + here->MOS2qbs);
            return(OK);
        case MOS2_CQBS:
            value->rValue = *(ckt->CKTstate0 + here->MOS2cqbs);
            return(OK);
        case MOS2_L_SENS_DC:
            if(ckt->CKTsenInfo){
               value->rValue = *(ckt->CKTsenInfo->SEN_Sap[select->iValue + 1]+
                here->MOS2senParmNo);
            }
            return(OK);
        case MOS2_L_SENS_REAL:
            if(ckt->CKTsenInfo){
               value->rValue = *(ckt->CKTsenInfo->SEN_RHS[select->iValue + 1]+
                here->MOS2senParmNo);
            }
            return(OK);
        case MOS2_L_SENS_IMAG:
            if(ckt->CKTsenInfo){
               value->rValue = *(ckt->CKTsenInfo->SEN_iRHS[select->iValue + 1]+
                here->MOS2senParmNo);
            }
            return(OK);
        case MOS2_L_SENS_MAG:
            if(ckt->CKTsenInfo){
                vr = *(ckt->CKTrhsOld + select->iValue + 1); 
                vi = *(ckt->CKTirhsOld + select->iValue + 1); 
                vm = sqrt(vr*vr + vi*vi);
                if(vm == 0){
                    value->rValue = 0;
                    return(OK);
                }
                sr = *(ckt->CKTsenInfo->SEN_RHS[select->iValue + 1]+
                        here->MOS2senParmNo);
                si = *(ckt->CKTsenInfo->SEN_iRHS[select->iValue + 1]+
                        here->MOS2senParmNo);
                value->rValue = (vr * sr + vi * si)/vm;
            }
            return(OK);
        case MOS2_L_SENS_PH:
            if(ckt->CKTsenInfo){
                vr = *(ckt->CKTrhsOld + select->iValue + 1); 
                vi = *(ckt->CKTirhsOld + select->iValue + 1); 
                vm = vr*vr + vi*vi;
                if(vm == 0){
                    value->rValue = 0;
                    return(OK);
                }
                sr = *(ckt->CKTsenInfo->SEN_RHS[select->iValue + 1]+
                        here->MOS2senParmNo);
                si = *(ckt->CKTsenInfo->SEN_iRHS[select->iValue + 1]+
                    here->MOS2senParmNo);
                value->rValue =  (vr * si - vi * sr)/vm;
            }
            return(OK);
        case MOS2_L_SENS_CPLX:
            if(ckt->CKTsenInfo){
                value->cValue.real= 
                        *(ckt->CKTsenInfo->SEN_RHS[select->iValue + 1]+
                        here->MOS2senParmNo);
                value->cValue.imag= 
                        *(ckt->CKTsenInfo->SEN_iRHS[select->iValue + 1]+
                        here->MOS2senParmNo);
            }
            return(OK);
        case MOS2_W_SENS_DC:
            if(ckt->CKTsenInfo){
                value->rValue = *(ckt->CKTsenInfo->SEN_Sap[select->iValue + 1]+
                        here->MOS2senParmNo + here->MOS2sens_l);
            }
            return(OK);
        case MOS2_W_SENS_REAL:
            if(ckt->CKTsenInfo){
                value->rValue = *(ckt->CKTsenInfo->SEN_RHS[select->iValue + 1]+
                        here->MOS2senParmNo + here->MOS2sens_l);
            }
            return(OK);
        case MOS2_W_SENS_IMAG:
            if(ckt->CKTsenInfo){
                value->rValue = *(ckt->CKTsenInfo->SEN_iRHS[select->iValue + 1]+
                        here->MOS2senParmNo + here->MOS2sens_l);
            }
            return(OK);
        case MOS2_W_SENS_MAG:
            if(ckt->CKTsenInfo){
                vr = *(ckt->CKTrhsOld + select->iValue + 1); 
                vi = *(ckt->CKTirhsOld + select->iValue + 1); 
                vm = sqrt(vr*vr + vi*vi);
                if(vm == 0){
                    value->rValue = 0;
                    return(OK);
                }
                sr = *(ckt->CKTsenInfo->SEN_RHS[select->iValue + 1]+
                        here->MOS2senParmNo + here->MOS2sens_l);
                si = *(ckt->CKTsenInfo->SEN_iRHS[select->iValue + 1]+
                        here->MOS2senParmNo + here->MOS2sens_l);
                        value->rValue = (vr * sr + vi * si)/vm;
            }
            return(OK);
        case MOS2_W_SENS_PH:
            if(ckt->CKTsenInfo){
                vr = *(ckt->CKTrhsOld + select->iValue + 1); 
                vi = *(ckt->CKTirhsOld + select->iValue + 1); 
                vm = vr*vr + vi*vi;
                if(vm == 0){
                    value->rValue = 0;
                    return(OK);
                }
                sr = *(ckt->CKTsenInfo->SEN_RHS[select->iValue + 1]+
                        here->MOS2senParmNo + here->MOS2sens_l);
                si = *(ckt->CKTsenInfo->SEN_iRHS[select->iValue + 1]+
                        here->MOS2senParmNo + here->MOS2sens_l);
                value->rValue =  (vr * si - vi * sr)/vm;
            }
            return(OK);
        case MOS2_W_SENS_CPLX:
            if(ckt->CKTsenInfo){
                value->cValue.real= 
                        *(ckt->CKTsenInfo->SEN_RHS[select->iValue + 1]+
                        here->MOS2senParmNo + here->MOS2sens_l);
                value->cValue.imag= 
                        *(ckt->CKTsenInfo->SEN_iRHS[select->iValue + 1]+
                        here->MOS2senParmNo + here->MOS2sens_l);
             }
             return(OK);
        case MOS2_CB :
            if (ckt->CKTcurrentAnalysis & DOING_AC) {
                errMsg = TMALLOC(char, strlen(msg) + 1);
                errRtn = "MOS2ask.c";
                strcpy(errMsg,msg);
                return(E_ASKCURRENT);
            } else {
                value->rValue = here->MOS2cbd + here->MOS2cbs - *(ckt->CKTstate0
                        + here->MOS2cqgb);
            }
            return(OK);
        case MOS2_CG :
            if (ckt->CKTcurrentAnalysis & DOING_AC) {
                errMsg = TMALLOC(char, strlen(msg) + 1);
                errRtn = "MOS2ask.c";
                strcpy(errMsg,msg);
                return(E_ASKCURRENT);
            } else if (ckt->CKTcurrentAnalysis & (DOING_DCOP | DOING_TRCV)) {
                value->rValue = 0;
            } else if ((ckt->CKTcurrentAnalysis & DOING_TRAN) && 
                   (ckt->CKTmode & MODETRANOP)) {
                value->rValue = 0;
            } else {
                value->rValue = *(ckt->CKTstate0 + here->MOS2cqgb) +
                        *(ckt->CKTstate0 + here->MOS2cqgd) + *(ckt->CKTstate0 + 
                        here->MOS2cqgs);
            }
            return(OK);
        case MOS2_CS :
            if (ckt->CKTcurrentAnalysis & DOING_AC) {
                errMsg = TMALLOC(char, strlen(msg) + 1);
                errRtn = "MOS2ask.c";
                strcpy(errMsg,msg);
                return(E_ASKCURRENT);
            } else {
                value->rValue = -here->MOS2cd;
                value->rValue -= here->MOS2cbd + here->MOS2cbs -
                        *(ckt->CKTstate0 + here->MOS2cqgb);
                if ((ckt->CKTcurrentAnalysis & DOING_TRAN) && 
                        !(ckt->CKTmode & MODETRANOP)) {
                    value->rValue -= *(ckt->CKTstate0 + here->MOS2cqgb) + 
                            *(ckt->CKTstate0 + here->MOS2cqgd) + 
                            *(ckt->CKTstate0 + here->MOS2cqgs);
                }
            }
            return(OK);
        case MOS2_POWER :
            if (ckt->CKTcurrentAnalysis & DOING_AC) {
                errMsg = TMALLOC(char, strlen(msg) + 1); 
                errRtn = "MOS2ask.c";
                strcpy(errMsg,msg);
                return(E_ASKPOWER);
            } else {
                double temp;

                value->rValue = here->MOS2cd * 
                        *(ckt->CKTrhsOld + here->MOS2dNode);
                value->rValue += (here->MOS2cbd + here->MOS2cbs -
                        *(ckt->CKTstate0 + here->MOS2cqgb)) * 
                        *(ckt->CKTrhsOld + here->MOS2bNode);
                if ((ckt->CKTcurrentAnalysis & DOING_TRAN) && 
                        !(ckt->CKTmode & MODETRANOP)) {
                    value->rValue += (*(ckt->CKTstate0 + here->MOS2cqgb) + 
                            *(ckt->CKTstate0 + here->MOS2cqgd) + 
                            *(ckt->CKTstate0 + here->MOS2cqgs)) *
                            *(ckt->CKTrhsOld + here->MOS2gNode);
                }
                temp = -here->MOS2cd;
                temp -= here->MOS2cbd + here->MOS2cbs ;
                if ((ckt->CKTcurrentAnalysis & DOING_TRAN) && 
                        !(ckt->CKTmode & MODETRANOP)) {
                    temp -= *(ckt->CKTstate0 + here->MOS2cqgb) + 
                            *(ckt->CKTstate0 + here->MOS2cqgd) + 
                            *(ckt->CKTstate0 + here->MOS2cqgs);
                }
                value->rValue += temp * *(ckt->CKTrhsOld + here->MOS2sNode);
            }
            return(OK);
        default:
            return(E_BADPARM);
    }
    /* NOTREACHED */
}

