/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1987 Thomas L. Quarles
Modified: 2000 AlansFixes
**********/

#include "ngspice/ngspice.h"
#include "ngspice/const.h"
#include "ngspice/ifsim.h"
#include "ngspice/cktdefs.h"
#include "ngspice/devdefs.h"
#include "mos1defs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

/*ARGSUSED*/
int
MOS1ask(CKTcircuit *ckt, GENinstance *inst, int which, IFvalue *value,
        IFvalue *select)
{
    MOS1instance *here = (MOS1instance*)inst;
    double vr;
    double vi;
    double sr;
    double si;
    double vm;
    static char *msg = "Current and power not available for ac analysis";
    switch(which) {
        case MOS1_TEMP:
            value->rValue = here->MOS1temp - CONSTCtoK;
            return(OK);
        case MOS1_DTEMP:
            value->rValue = here->MOS1dtemp;
            return(OK);
        case MOS1_CGS:
            value->rValue = 2*  *(ckt->CKTstate0 + here->MOS1capgs);
            return(OK);
        case MOS1_CGD:
            value->rValue = 2* *(ckt->CKTstate0 + here->MOS1capgd);
            return(OK);   
        case MOS1_M:
            value->rValue = here->MOS1m;
            return(OK);    
        case MOS1_L:
            value->rValue = here->MOS1l;
                return(OK);
        case MOS1_W:
            value->rValue = here->MOS1w;
                return(OK);
        case MOS1_AS:
            value->rValue = here->MOS1sourceArea;
                return(OK);
        case MOS1_AD:
            value->rValue = here->MOS1drainArea;
                return(OK);
        case MOS1_PS:
            value->rValue = here->MOS1sourcePerimiter;
                return(OK);
        case MOS1_PD:
            value->rValue = here->MOS1drainPerimiter;
                return(OK);
        case MOS1_NRS:
            value->rValue = here->MOS1sourceSquares;
                return(OK);
        case MOS1_NRD:
            value->rValue = here->MOS1drainSquares;
                return(OK);
        case MOS1_OFF:
            value->rValue = here->MOS1off;
                return(OK);
        case MOS1_IC_VBS:
            value->rValue = here->MOS1icVBS;
                return(OK);
        case MOS1_IC_VDS:
            value->rValue = here->MOS1icVDS;
                return(OK);
        case MOS1_IC_VGS:
            value->rValue = here->MOS1icVGS;
                return(OK);
        case MOS1_DNODE:
            value->iValue = here->MOS1dNode;
            return(OK);
        case MOS1_GNODE:
            value->iValue = here->MOS1gNode;
            return(OK);
        case MOS1_SNODE:
            value->iValue = here->MOS1sNode;
            return(OK);
        case MOS1_BNODE:
            value->iValue = here->MOS1bNode;
            return(OK);
        case MOS1_DNODEPRIME:
            value->iValue = here->MOS1dNodePrime;
            return(OK);
        case MOS1_SNODEPRIME:
            value->iValue = here->MOS1sNodePrime;
            return(OK);
        case MOS1_SOURCECONDUCT:
            value->rValue = here->MOS1sourceConductance;
            return(OK);
        case MOS1_SOURCERESIST:
	    if (here->MOS1sNodePrime != here->MOS1sNode)
		value->rValue = 1.0 / here->MOS1sourceConductance;
	    else
		value->rValue = 0.0;
            return(OK);
        case MOS1_DRAINCONDUCT:
            value->rValue = here->MOS1drainConductance;
            return(OK);
        case MOS1_DRAINRESIST:
	    if (here->MOS1dNodePrime != here->MOS1dNode)
		value->rValue = 1.0 / here->MOS1drainConductance;
	    else
		value->rValue = 0.0;
            return(OK);
        case MOS1_VON:
            value->rValue = here->MOS1von;
            return(OK);
        case MOS1_VDSAT:
            value->rValue = here->MOS1vdsat;
            return(OK);
        case MOS1_SOURCEVCRIT:
            value->rValue = here->MOS1sourceVcrit;
            return(OK);
        case MOS1_DRAINVCRIT:
            value->rValue = here->MOS1drainVcrit;
            return(OK);
        case MOS1_CD:
            value->rValue = here->MOS1cd;
            return(OK);
        case MOS1_CBS:
            value->rValue = here->MOS1cbs;
            return(OK);
        case MOS1_CBD:
            value->rValue = here->MOS1cbd;
            return(OK);
        case MOS1_GMBS:
            value->rValue = here->MOS1gmbs;
            return(OK);
        case MOS1_GM:
            value->rValue = here->MOS1gm;
            return(OK);
        case MOS1_GDS:
            value->rValue = here->MOS1gds;
            return(OK);
        case MOS1_GBD:
            value->rValue = here->MOS1gbd;
            return(OK);
        case MOS1_GBS:
            value->rValue = here->MOS1gbs;
            return(OK);
        case MOS1_CAPBD:
            value->rValue = here->MOS1capbd;
            return(OK);
        case MOS1_CAPBS:
            value->rValue = here->MOS1capbs;
            return(OK);
        case MOS1_CAPZEROBIASBD:
            value->rValue = here->MOS1Cbd;
            return(OK);
        case MOS1_CAPZEROBIASBDSW:
            value->rValue = here->MOS1Cbdsw;
            return(OK);
        case MOS1_CAPZEROBIASBS:
            value->rValue = here->MOS1Cbs;
            return(OK);
        case MOS1_CAPZEROBIASBSSW:
            value->rValue = here->MOS1Cbssw;
            return(OK);
        case MOS1_VBD:
            value->rValue = *(ckt->CKTstate0 + here->MOS1vbd);
            return(OK);
        case MOS1_VBS:
            value->rValue = *(ckt->CKTstate0 + here->MOS1vbs);
            return(OK);
        case MOS1_VGS:
            value->rValue = *(ckt->CKTstate0 + here->MOS1vgs);
            return(OK);
        case MOS1_VDS:
            value->rValue = *(ckt->CKTstate0 + here->MOS1vds);
            return(OK);
        case MOS1_CAPGS:
            value->rValue = 2* *(ckt->CKTstate0 + here->MOS1capgs);
            /* add overlap capacitance */
            value->rValue += (MOS1modPtr(here)->MOS1gateSourceOverlapCapFactor)
                             * here->MOS1m
                             * (here->MOS1w);
            return(OK);
        case MOS1_QGS:
            value->rValue = *(ckt->CKTstate0 + here->MOS1qgs);
            return(OK);
        case MOS1_CQGS:
            value->rValue = *(ckt->CKTstate0 + here->MOS1cqgs);
            return(OK);
        case MOS1_CAPGD:
            value->rValue = 2* *(ckt->CKTstate0 + here->MOS1capgd);
            /* add overlap capacitance */
            value->rValue += (MOS1modPtr(here)->MOS1gateDrainOverlapCapFactor)
                             * here->MOS1m
                             * (here->MOS1w);
            return(OK);
        case MOS1_QGD:
            value->rValue = *(ckt->CKTstate0 + here->MOS1qgd);
            return(OK);
        case MOS1_CQGD:
            value->rValue = *(ckt->CKTstate0 + here->MOS1cqgd);
            return(OK);
        case MOS1_CAPGB:
            value->rValue = 2* *(ckt->CKTstate0 + here->MOS1capgb);
            /* add overlap capacitance */
            value->rValue += (MOS1modPtr(here)->MOS1gateBulkOverlapCapFactor)
                             * here->MOS1m
                             * (here->MOS1l
                                -2*(MOS1modPtr(here)->MOS1latDiff));
            return(OK);
        case MOS1_QGB:
            value->rValue = *(ckt->CKTstate0 + here->MOS1qgb);
            return(OK);
        case MOS1_CQGB:
            value->rValue = *(ckt->CKTstate0 + here->MOS1cqgb);
            return(OK);
        case MOS1_QBD:
            value->rValue = *(ckt->CKTstate0 + here->MOS1qbd);
            return(OK);
        case MOS1_CQBD:
            value->rValue = *(ckt->CKTstate0 + here->MOS1cqbd);
            return(OK);
        case MOS1_QBS:
            value->rValue = *(ckt->CKTstate0 + here->MOS1qbs);
            return(OK);
        case MOS1_CQBS:
            value->rValue = *(ckt->CKTstate0 + here->MOS1cqbs);
            return(OK);
        case MOS1_L_SENS_DC:
            if(ckt->CKTsenInfo && here->MOS1sens_l){
               value->rValue = *(ckt->CKTsenInfo->SEN_Sap[select->iValue + 1]+
                       here->MOS1senParmNo);
            }
            return(OK);
        case MOS1_L_SENS_REAL:
            if(ckt->CKTsenInfo && here->MOS1sens_l){
               value->rValue = *(ckt->CKTsenInfo->SEN_RHS[select->iValue + 1]+
                       here->MOS1senParmNo);
            }
            return(OK);
        case MOS1_L_SENS_IMAG:
            if(ckt->CKTsenInfo && here->MOS1sens_l){
               value->rValue = *(ckt->CKTsenInfo->SEN_iRHS[select->iValue + 1]+
                       here->MOS1senParmNo);
            }
            return(OK);
        case MOS1_L_SENS_MAG:
            if(ckt->CKTsenInfo && here->MOS1sens_l){
                vr = *(ckt->CKTrhsOld + select->iValue + 1); 
                vi = *(ckt->CKTirhsOld + select->iValue + 1); 
                vm = sqrt(vr*vr + vi*vi);
                if(vm == 0){
                    value->rValue = 0;
                    return(OK);
                }
                sr = *(ckt->CKTsenInfo->SEN_RHS[select->iValue + 1]+
                        here->MOS1senParmNo);
                si = *(ckt->CKTsenInfo->SEN_iRHS[select->iValue + 1]+
                        here->MOS1senParmNo);
                value->rValue = (vr * sr + vi * si)/vm;
            }
            return(OK);
        case MOS1_L_SENS_PH:
            if(ckt->CKTsenInfo && here->MOS1sens_l){
                vr = *(ckt->CKTrhsOld + select->iValue + 1); 
                vi = *(ckt->CKTirhsOld + select->iValue + 1); 
                vm = vr*vr + vi*vi;
                if(vm == 0){
                    value->rValue = 0;
                    return(OK);
                }
                sr = *(ckt->CKTsenInfo->SEN_RHS[select->iValue + 1]+
                        here->MOS1senParmNo);
                si = *(ckt->CKTsenInfo->SEN_iRHS[select->iValue + 1]+
                        here->MOS1senParmNo);
                value->rValue =  (vr * si - vi * sr)/vm;
            }
            return(OK);
        case MOS1_L_SENS_CPLX:
            if(ckt->CKTsenInfo && here->MOS1sens_l){
                value->cValue.real= 
                        *(ckt->CKTsenInfo->SEN_RHS[select->iValue + 1]+
                        here->MOS1senParmNo);
                value->cValue.imag= 
                        *(ckt->CKTsenInfo->SEN_iRHS[select->iValue + 1]+
                        here->MOS1senParmNo);
            }
            return(OK);
        case MOS1_W_SENS_DC:
            if(ckt->CKTsenInfo && here->MOS1sens_w){
                value->rValue = *(ckt->CKTsenInfo->SEN_Sap[select->iValue + 1]+
                        here->MOS1senParmNo + here->MOS1sens_l);
            }
            return(OK);
        case MOS1_W_SENS_REAL:
            if(ckt->CKTsenInfo && here->MOS1sens_w){
                value->rValue = *(ckt->CKTsenInfo->SEN_RHS[select->iValue + 1]+
                        here->MOS1senParmNo + here->MOS1sens_l);
            }
             return(OK);
        case MOS1_W_SENS_IMAG:
            if(ckt->CKTsenInfo && here->MOS1sens_w){
                value->rValue = *(ckt->CKTsenInfo->SEN_iRHS[select->iValue + 1]+
                        here->MOS1senParmNo + here->MOS1sens_l);
            }
            return(OK);
        case MOS1_W_SENS_MAG:
            if(ckt->CKTsenInfo && here->MOS1sens_w){
                vr = *(ckt->CKTrhsOld + select->iValue + 1); 
                vi = *(ckt->CKTirhsOld + select->iValue + 1); 
                vm = sqrt(vr*vr + vi*vi);
                if(vm == 0){
                    value->rValue = 0;
                    return(OK);
                }
                sr = *(ckt->CKTsenInfo->SEN_RHS[select->iValue + 1]+
                        here->MOS1senParmNo + here->MOS1sens_l);
                si = *(ckt->CKTsenInfo->SEN_iRHS[select->iValue + 1]+
                        here->MOS1senParmNo + here->MOS1sens_l);
                value->rValue = (vr * sr + vi * si)/vm;
            }
            return(OK);
        case MOS1_W_SENS_PH:
            if(ckt->CKTsenInfo && here->MOS1sens_w){
                vr = *(ckt->CKTrhsOld + select->iValue + 1); 
                vi = *(ckt->CKTirhsOld + select->iValue + 1);     
                vm = vr*vr + vi*vi;
                if(vm == 0){
                    value->rValue = 0;
                    return(OK);
                }
                sr = *(ckt->CKTsenInfo->SEN_RHS[select->iValue + 1]+
                        here->MOS1senParmNo + here->MOS1sens_l);
                si = *(ckt->CKTsenInfo->SEN_iRHS[select->iValue + 1]+
                        here->MOS1senParmNo + here->MOS1sens_l);
                value->rValue =  (vr * si - vi * sr)/vm;
            }
                    return(OK);
        case MOS1_W_SENS_CPLX:
            if(ckt->CKTsenInfo && here->MOS1sens_w){
                value->cValue.real= 
                        *(ckt->CKTsenInfo->SEN_RHS[select->iValue + 1]+
                        here->MOS1senParmNo + here->MOS1sens_l);
                value->cValue.imag= 
                        *(ckt->CKTsenInfo->SEN_iRHS[select->iValue + 1]+
                        here->MOS1senParmNo + here->MOS1sens_l);
            }
            return(OK);
        case MOS1_CB :
            if (ckt->CKTcurrentAnalysis & DOING_AC) {
                errMsg = TMALLOC(char, strlen(msg) + 1);
                errRtn = "MOS1ask.c";
                strcpy(errMsg,msg);
                return(E_ASKCURRENT);
            } else {
                value->rValue = here->MOS1cbd + here->MOS1cbs - *(ckt->CKTstate0
                        + here->MOS1cqgb);
            }
            return(OK);
        case MOS1_CG :
            if (ckt->CKTcurrentAnalysis & DOING_AC) {
                errMsg = TMALLOC(char, strlen(msg) + 1);
                errRtn = "MOS1ask.c";
                strcpy(errMsg,msg);
                return(E_ASKCURRENT);
            } else if (ckt->CKTcurrentAnalysis & (DOING_DCOP | DOING_TRCV)) {
                value->rValue = 0;
            } else if ((ckt->CKTcurrentAnalysis & DOING_TRAN) && 
                    (ckt->CKTmode & MODETRANOP)) {
                value->rValue = 0;
            } else {
                value->rValue = *(ckt->CKTstate0 + here->MOS1cqgb) +
                        *(ckt->CKTstate0 + here->MOS1cqgd) + *(ckt->CKTstate0 + 
                        here->MOS1cqgs);
            }
            return(OK);
        case MOS1_CS :
            if (ckt->CKTcurrentAnalysis & DOING_AC) {
                errMsg = TMALLOC(char, strlen(msg) + 1);
                errRtn = "MOS1ask.c";
                strcpy(errMsg,msg);
                return(E_ASKCURRENT);
            } else {
                value->rValue = -here->MOS1cd;
                value->rValue -= here->MOS1cbd + here->MOS1cbs -
                        *(ckt->CKTstate0 + here->MOS1cqgb);
                if ((ckt->CKTcurrentAnalysis & DOING_TRAN) && 
                        !(ckt->CKTmode & MODETRANOP)) {
                    value->rValue -= *(ckt->CKTstate0 + here->MOS1cqgb) + 
                            *(ckt->CKTstate0 + here->MOS1cqgd) +
                            *(ckt->CKTstate0 + here->MOS1cqgs);
                }
            }
            return(OK);
        case MOS1_POWER :
            if (ckt->CKTcurrentAnalysis & DOING_AC) {
                errMsg = TMALLOC(char, strlen(msg) + 1);
                errRtn = "MOS1ask.c";
                strcpy(errMsg,msg);
                return(E_ASKPOWER);
            } else {
                double temp;

                value->rValue = here->MOS1cd * 
                        *(ckt->CKTrhsOld + here->MOS1dNode);
                value->rValue += (here->MOS1cbd + here->MOS1cbs -
                        *(ckt->CKTstate0 + here->MOS1cqgb)) *
                        *(ckt->CKTrhsOld + here->MOS1bNode);
                if ((ckt->CKTcurrentAnalysis & DOING_TRAN) && 
                        !(ckt->CKTmode & MODETRANOP)) {
                    value->rValue += (*(ckt->CKTstate0 + here->MOS1cqgb) + 
                            *(ckt->CKTstate0 + here->MOS1cqgd) +
                            *(ckt->CKTstate0 + here->MOS1cqgs)) *
                            *(ckt->CKTrhsOld + here->MOS1gNode);
                }
                temp = -here->MOS1cd;
                temp -= here->MOS1cbd + here->MOS1cbs ;
                if ((ckt->CKTcurrentAnalysis & DOING_TRAN) && 
                        !(ckt->CKTmode & MODETRANOP)) {
                    temp -= *(ckt->CKTstate0 + here->MOS1cqgb) + 
                            *(ckt->CKTstate0 + here->MOS1cqgd) + 
                            *(ckt->CKTstate0 + here->MOS1cqgs);
                }
                value->rValue += temp * *(ckt->CKTrhsOld + here->MOS1sNode);
            }
            return(OK);
        default:
            return(E_BADPARM);
    }
    /* NOTREACHED */
}

