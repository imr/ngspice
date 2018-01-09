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
#include "mos3defs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

/*ARGSUSED*/
int
MOS3ask(CKTcircuit *ckt, GENinstance *inst, int which, IFvalue *value,
        IFvalue *select)
{
    MOS3instance *here = (MOS3instance *)inst;
    double vr;
    double vi;
    double sr;
    double si;
    double vm;
    static char *msg = "Current and power not available for ac analysis";
    switch(which) {
        case MOS3_TEMP:
            value->rValue = here->MOS3temp-CONSTCtoK;
            return(OK);
        case MOS3_DTEMP:
            value->rValue = here->MOS3dtemp;
            return(OK);
        case MOS3_CGS:
            value->rValue = 2* *(ckt->CKTstate0 + here->MOS3capgs);
            return(OK);
        case MOS3_CGD:
            value->rValue = 2* *(ckt->CKTstate0 + here->MOS3capgd);
            return(OK);
        case MOS3_M:
            value->rValue = here->MOS3m;
            return(OK);
        case MOS3_L:
            value->rValue = here->MOS3l;
                return(OK);
        case MOS3_W:
            value->rValue = here->MOS3w;
                return(OK);
        case MOS3_AS:
            value->rValue = here->MOS3sourceArea;
                return(OK);
        case MOS3_AD:
            value->rValue = here->MOS3drainArea;
                return(OK);
        case MOS3_PS:
            value->rValue = here->MOS3sourcePerimiter;
                return(OK);
        case MOS3_PD:
            value->rValue = here->MOS3drainPerimiter;
                return(OK);
        case MOS3_NRS:
            value->rValue = here->MOS3sourceSquares;
                return(OK);
        case MOS3_NRD:
            value->rValue = here->MOS3drainSquares;
                return(OK);
        case MOS3_OFF:
            value->rValue = here->MOS3off;
                return(OK);
        case MOS3_IC_VBS:
            value->rValue = here->MOS3icVBS;
                return(OK);
        case MOS3_IC_VDS:
            value->rValue = here->MOS3icVDS;
                return(OK);
        case MOS3_IC_VGS:
            value->rValue = here->MOS3icVGS;
            return(OK);
        case MOS3_DNODE:
            value->iValue = here->MOS3dNode;
            return(OK);
        case MOS3_GNODE:
            value->iValue = here->MOS3gNode;
            return(OK);
        case MOS3_SNODE:
            value->iValue = here->MOS3sNode;
            return(OK);
        case MOS3_BNODE:
            value->iValue = here->MOS3bNode;
            return(OK);
        case MOS3_DNODEPRIME:
            value->iValue = here->MOS3dNodePrime;
            return(OK);
        case MOS3_SNODEPRIME:
            value->iValue = here->MOS3sNodePrime;
            return(OK);
        case MOS3_SOURCECONDUCT:
	    value->rValue = here->MOS3sourceConductance;
            return(OK);
        case MOS3_DRAINCONDUCT:
	    value->rValue = here->MOS3drainConductance;
            return(OK);
        case MOS3_SOURCERESIST:
            if (here->MOS3sNodePrime != here->MOS3sNode)
		value->rValue = 1.0 / here->MOS3sourceConductance;
	    else
		value->rValue = 0.0;
            return(OK);
        case MOS3_DRAINRESIST:
            if (here->MOS3dNodePrime != here->MOS3dNode)
		value->rValue = 1.0 / here->MOS3drainConductance;
	    else
		value->rValue = 0.0;
            return(OK);
        case MOS3_VON:
            value->rValue = here->MOS3von;
            return(OK);
        case MOS3_VDSAT:
            value->rValue = here->MOS3vdsat;
            return(OK);
        case MOS3_SOURCEVCRIT:
            value->rValue = here->MOS3sourceVcrit;
            return(OK);
        case MOS3_DRAINVCRIT:
            value->rValue = here->MOS3drainVcrit;
            return(OK);
        case MOS3_CD:
            value->rValue = here->MOS3cd;
            return(OK);
        case MOS3_CBS:
            value->rValue = here->MOS3cbs;
            return(OK);
        case MOS3_CBD:
            value->rValue = here->MOS3cbd;
            return(OK);
        case MOS3_GMBS:
            value->rValue = here->MOS3gmbs;
            return(OK);
        case MOS3_GM:
            value->rValue = here->MOS3gm;
            return(OK);
        case MOS3_GDS:
            value->rValue = here->MOS3gds;
            return(OK);
        case MOS3_GBD:
            value->rValue = here->MOS3gbd;
            return(OK);
        case MOS3_GBS:
            value->rValue = here->MOS3gbs;
            return(OK);
        case MOS3_CAPBD:
            value->rValue = here->MOS3capbd;
            return(OK);
        case MOS3_CAPBS:
            value->rValue = here->MOS3capbs;
            return(OK);
        case MOS3_CAPZEROBIASBD:
            value->rValue = here->MOS3Cbd;
            return(OK);
        case MOS3_CAPZEROBIASBDSW:
            value->rValue = here->MOS3Cbdsw;
            return(OK);
        case MOS3_CAPZEROBIASBS:
            value->rValue = here->MOS3Cbs;
            return(OK);
        case MOS3_CAPZEROBIASBSSW:
            value->rValue = here->MOS3Cbssw;
            return(OK);
        case MOS3_VBD:
            value->rValue = *(ckt->CKTstate0 + here->MOS3vbd);
            return(OK);
        case MOS3_VBS:
            value->rValue = *(ckt->CKTstate0 + here->MOS3vbs);
            return(OK);
        case MOS3_VGS:
            value->rValue = *(ckt->CKTstate0 + here->MOS3vgs);
            return(OK);
        case MOS3_VDS:
            value->rValue = *(ckt->CKTstate0 + here->MOS3vds);
            return(OK);
        case MOS3_CAPGS:
            value->rValue = 2* *(ckt->CKTstate0 + here->MOS3capgs);
/* add overlap capacitance */
            value->rValue += (MOS3modPtr(here)->MOS3gateSourceOverlapCapFactor)
                             * here->MOS3m
                             * (here->MOS3w
                                +MOS3modPtr(here)->MOS3widthAdjust
                                -2*(MOS3modPtr(here)->MOS3widthNarrow));
            return(OK);
        case MOS3_QGS:
            value->rValue = *(ckt->CKTstate0 + here->MOS3qgs);
            return(OK);
        case MOS3_CQGS:
            value->rValue = *(ckt->CKTstate0 + here->MOS3cqgs);
            return(OK);
        case MOS3_CAPGD:
             value->rValue = 2* *(ckt->CKTstate0 + here->MOS3capgd);
/* add overlap capacitance */
            value->rValue += (MOS3modPtr(here)->MOS3gateDrainOverlapCapFactor)
                             * here->MOS3m
                             * (here->MOS3w
                                +MOS3modPtr(here)->MOS3widthAdjust
                                -2*(MOS3modPtr(here)->MOS3widthNarrow));
            return(OK);
        case MOS3_QGD:
            value->rValue = *(ckt->CKTstate0 + here->MOS3qgd);
            return(OK);
        case MOS3_CQGD:
            value->rValue = *(ckt->CKTstate0 + here->MOS3cqgd);
            return(OK);
        case MOS3_CAPGB:
            value->rValue = 2* *(ckt->CKTstate0 + here->MOS3capgb);
/* add overlap capacitance */
            value->rValue += (MOS3modPtr(here)->MOS3gateBulkOverlapCapFactor)
                             * here->MOS3m
                             * (here->MOS3l
                                +MOS3modPtr(here)->MOS3lengthAdjust
                                -2*(MOS3modPtr(here)->MOS3latDiff));
            return(OK);
        case MOS3_QGB:
            value->rValue = *(ckt->CKTstate0 + here->MOS3qgb);
            return(OK);
        case MOS3_CQGB:
            value->rValue = *(ckt->CKTstate0 + here->MOS3cqgb);
            return(OK);
        case MOS3_QBD:
            value->rValue = *(ckt->CKTstate0 + here->MOS3qbd);
            return(OK);
        case MOS3_CQBD:
            value->rValue = *(ckt->CKTstate0 + here->MOS3cqbd);
            return(OK);
        case MOS3_QBS:
            value->rValue = *(ckt->CKTstate0 + here->MOS3qbs);
            return(OK);
        case MOS3_CQBS:
            value->rValue = *(ckt->CKTstate0 + here->MOS3cqbs);
            return(OK);
        case MOS3_L_SENS_DC:
            if(ckt->CKTsenInfo){
                value->rValue = *(ckt->CKTsenInfo->SEN_Sap[select->iValue + 1]+
                        here->MOS3senParmNo);
            }
            return(OK);
        case MOS3_L_SENS_REAL:
            if(ckt->CKTsenInfo){
            value->rValue = *(ckt->CKTsenInfo->SEN_RHS[select->iValue + 1]+
                        here->MOS3senParmNo);
            }
            return(OK);
        case MOS3_L_SENS_IMAG:
            if(ckt->CKTsenInfo){
                value->rValue = *(ckt->CKTsenInfo->SEN_iRHS[select->iValue + 1]+
                        here->MOS3senParmNo);
            }
            return(OK);
        case MOS3_L_SENS_MAG:
            if(ckt->CKTsenInfo){
                vr = *(ckt->CKTrhsOld + select->iValue + 1); 
                vi = *(ckt->CKTirhsOld + select->iValue + 1); 
                vm = sqrt(vr*vr + vi*vi);
                if(vm == 0){
                    value->rValue = 0;
                    return(OK);
                }
                sr = *(ckt->CKTsenInfo->SEN_RHS[select->iValue + 1]+
                        here->MOS3senParmNo);
                si = *(ckt->CKTsenInfo->SEN_iRHS[select->iValue + 1]+
                        here->MOS3senParmNo);
                value->rValue = (vr * sr + vi * si)/vm;
            }
            return(OK);
        case MOS3_L_SENS_PH:
            if(ckt->CKTsenInfo){
                vr = *(ckt->CKTrhsOld + select->iValue + 1); 
                vi = *(ckt->CKTirhsOld + select->iValue + 1); 
                vm = vr*vr + vi*vi;
                if(vm == 0){
                    value->rValue = 0;
                    return(OK);
                }
                sr = *(ckt->CKTsenInfo->SEN_RHS[select->iValue + 1]+
                        here->MOS3senParmNo);
                si = *(ckt->CKTsenInfo->SEN_iRHS[select->iValue + 1]+
                        here->MOS3senParmNo);
                value->rValue =  (vr * si - vi * sr)/vm;
            }
            return(OK);
        case MOS3_L_SENS_CPLX:
            if(ckt->CKTsenInfo){
                value->cValue.real= 
                        *(ckt->CKTsenInfo->SEN_RHS[select->iValue + 1]+
                        here->MOS3senParmNo);
                value->cValue.imag= 
                        *(ckt->CKTsenInfo->SEN_iRHS[select->iValue + 1]+
                        here->MOS3senParmNo);
            }
            return(OK);
        case MOS3_W_SENS_DC:
            if(ckt->CKTsenInfo){
                value->rValue = *(ckt->CKTsenInfo->SEN_Sap[select->iValue + 1]+
                        here->MOS3senParmNo + here->MOS3sens_l);
            }
            return(OK);
        case MOS3_W_SENS_REAL:
            if(ckt->CKTsenInfo){
                value->rValue = *(ckt->CKTsenInfo->SEN_RHS[select->iValue + 1]+
                        here->MOS3senParmNo + here->MOS3sens_l);
            }
            return(OK);
        case MOS3_W_SENS_IMAG:
            if(ckt->CKTsenInfo){
                value->rValue = *(ckt->CKTsenInfo->SEN_iRHS[select->iValue + 1]+
                        here->MOS3senParmNo + here->MOS3sens_l);
            }
            return(OK);
        case MOS3_W_SENS_MAG:
            if(ckt->CKTsenInfo){
                vr = *(ckt->CKTrhsOld + select->iValue + 1); 
                vi = *(ckt->CKTirhsOld + select->iValue + 1); 
                vm = sqrt(vr*vr + vi*vi);
                if(vm == 0){
                    value->rValue = 0;
                    return(OK);
                }
                sr = *(ckt->CKTsenInfo->SEN_RHS[select->iValue + 1]+
                        here->MOS3senParmNo + here->MOS3sens_l);
                si = *(ckt->CKTsenInfo->SEN_iRHS[select->iValue + 1]+
                        here->MOS3senParmNo + here->MOS3sens_l);
                value->rValue = (vr * sr + vi * si)/vm;
            }
            return(OK);
        case MOS3_W_SENS_PH:
            if(ckt->CKTsenInfo){
                vr = *(ckt->CKTrhsOld + select->iValue + 1); 
                vi = *(ckt->CKTirhsOld + select->iValue + 1); 
                vm = vr*vr + vi*vi;
                if(vm == 0){
                    value->rValue = 0;
                    return(OK);
                }
                sr = *(ckt->CKTsenInfo->SEN_RHS[select->iValue + 1]+
                        here->MOS3senParmNo + here->MOS3sens_l);
                si = *(ckt->CKTsenInfo->SEN_iRHS[select->iValue + 1]+
                        here->MOS3senParmNo + here->MOS3sens_l);
                value->rValue =  (vr * si - vi * sr)/vm;
            }
            return(OK);
        case MOS3_W_SENS_CPLX:
            if(ckt->CKTsenInfo){
                value->cValue.real= 
                        *(ckt->CKTsenInfo->SEN_RHS[select->iValue + 1]+
                        here->MOS3senParmNo + here->MOS3sens_l);
                value->cValue.imag= 
                        *(ckt->CKTsenInfo->SEN_iRHS[select->iValue + 1]+
                        here->MOS3senParmNo + here->MOS3sens_l);
            }
            return(OK);
        case MOS3_CB :
            if (ckt->CKTcurrentAnalysis & DOING_AC) {
                errMsg = TMALLOC(char, strlen(msg) + 1);
                errRtn = "MOS3ask.c";
                strcpy(errMsg,msg);
                return(E_ASKCURRENT);
            } else {
                value->rValue = here->MOS3cbd + here->MOS3cbs - *(ckt->CKTstate0
                        + here->MOS3cqgb);
            }
            return(OK);
        case MOS3_CG :
            if (ckt->CKTcurrentAnalysis & DOING_AC) {
                errMsg = TMALLOC(char, strlen(msg) + 1);
                errRtn = "MOS3ask.c";
                strcpy(errMsg,msg);
                return(E_ASKCURRENT);
            } else if (ckt->CKTcurrentAnalysis & (DOING_DCOP | DOING_TRCV)) {
                value->rValue = 0;
            } else if ((ckt->CKTcurrentAnalysis & DOING_TRAN) && 
                        (ckt->CKTmode & MODETRANOP)) {
                value->rValue = 0;
            } else {
                value->rValue = *(ckt->CKTstate0 + here->MOS3cqgb) +
                    *(ckt->CKTstate0 + here->MOS3cqgd) + *(ckt->CKTstate0 + 
                    here->MOS3cqgs);
            }
            return(OK);
        case MOS3_CS :
            if (ckt->CKTcurrentAnalysis & DOING_AC) {
                errMsg = TMALLOC(char, strlen(msg) + 1);
                errRtn = "MOS3ask.c";
                strcpy(errMsg,msg);
                return(E_ASKCURRENT);
            } else {
                value->rValue = -here->MOS3cd;
                value->rValue -= here->MOS3cbd + here->MOS3cbs -
                        *(ckt->CKTstate0 + here->MOS3cqgb);
                if ((ckt->CKTcurrentAnalysis & DOING_TRAN) && 
                        !(ckt->CKTmode & MODETRANOP)) {
                    value->rValue -= *(ckt->CKTstate0 + here->MOS3cqgb) + 
                            *(ckt->CKTstate0 + here->MOS3cqgd) +
                            *(ckt->CKTstate0 + here->MOS3cqgs);
                }
            }
            return(OK);
        case MOS3_POWER :
            if (ckt->CKTcurrentAnalysis & DOING_AC) {
                errMsg = TMALLOC(char, strlen(msg) + 1);
                errRtn = "MOS3ask.c";
                strcpy(errMsg,msg);
                return(E_ASKPOWER);
            } else {
                double temp;

                value->rValue = here->MOS3cd * 
                        *(ckt->CKTrhsOld + here->MOS3dNode);
                value->rValue += (here->MOS3cbd + here->MOS3cbs -
                        *(ckt->CKTstate0 + here->MOS3cqgb)) *
                        *(ckt->CKTrhsOld + here->MOS3bNode);
                if ((ckt->CKTcurrentAnalysis & DOING_TRAN) && 
                        !(ckt->CKTmode & MODETRANOP)) {
                    value->rValue += (*(ckt->CKTstate0 + here->MOS3cqgb) + 
                            *(ckt->CKTstate0 + here->MOS3cqgd) + 
                            *(ckt->CKTstate0 + here->MOS3cqgs)) *
                            *(ckt->CKTrhsOld + here->MOS3gNode);
                }
                temp = -here->MOS3cd;
                temp -= here->MOS3cbd + here->MOS3cbs ;
                if ((ckt->CKTcurrentAnalysis & DOING_TRAN) && 
                        !(ckt->CKTmode & MODETRANOP)) {
                    temp -= *(ckt->CKTstate0 + here->MOS3cqgb) + 
                            *(ckt->CKTstate0 + here->MOS3cqgd) + 
                            *(ckt->CKTstate0 + here->MOS3cqgs);
                }
                value->rValue += temp * *(ckt->CKTrhsOld + here->MOS3sNode);
            }
            return(OK);
        default:
            return(E_BADPARM);
    }
    /* NOTREACHED */
}

