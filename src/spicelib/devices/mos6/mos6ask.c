/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1989 Takayasu Sakurai
**********/

#include "ngspice.h"
#include <stdio.h>
#include "const.h"
#include "ifsim.h"
#include "cktdefs.h"
#include "devdefs.h"
#include "mos6defs.h"
#include "sperror.h"
#include "suffix.h"

/*ARGSUSED*/
int
MOS6ask(ckt,inst,which,value,select)
    CKTcircuit *ckt;
    GENinstance *inst;
    int which;
    IFvalue *value;
    IFvalue *select;
{
    MOS6instance *here = (MOS6instance*)inst;
    double vr;
    double vi;
    double sr;
    double si;
    double vm;
    static char *msg = "Current and power not available for ac analysis";
    switch(which) {
        case MOS6_TEMP:
            value->rValue = here->MOS6temp-CONSTCtoK;
            return(OK);
        case MOS6_CGS:
            value->rValue = *(ckt->CKTstate0 + here->MOS6capgs);
            return(OK);
        case MOS6_CGD:
            value->rValue = *(ckt->CKTstate0 + here->MOS6capgd);
            return(OK);
        case MOS6_L:
            value->rValue = here->MOS6l;
                return(OK);
        case MOS6_W:
            value->rValue = here->MOS6w;
                return(OK);
        case MOS6_AS:
            value->rValue = here->MOS6sourceArea;
                return(OK);
        case MOS6_AD:
            value->rValue = here->MOS6drainArea;
                return(OK);
        case MOS6_PS:
            value->rValue = here->MOS6sourcePerimiter;
                return(OK);
        case MOS6_PD:
            value->rValue = here->MOS6drainPerimiter;
                return(OK);
        case MOS6_NRS:
            value->rValue = here->MOS6sourceSquares;
                return(OK);
        case MOS6_NRD:
            value->rValue = here->MOS6drainSquares;
                return(OK);
        case MOS6_OFF:
            value->rValue = here->MOS6off;
                return(OK);
        case MOS6_IC_VBS:
            value->rValue = here->MOS6icVBS;
                return(OK);
        case MOS6_IC_VDS:
            value->rValue = here->MOS6icVDS;
                return(OK);
        case MOS6_IC_VGS:
            value->rValue = here->MOS6icVGS;
                return(OK);
        case MOS6_DNODE:
            value->iValue = here->MOS6dNode;
            return(OK);
        case MOS6_GNODE:
            value->iValue = here->MOS6gNode;
            return(OK);
        case MOS6_SNODE:
            value->iValue = here->MOS6sNode;
            return(OK);
        case MOS6_BNODE:
            value->iValue = here->MOS6bNode;
            return(OK);
        case MOS6_DNODEPRIME:
            value->iValue = here->MOS6dNodePrime;
            return(OK);
        case MOS6_SNODEPRIME:
            value->iValue = here->MOS6sNodePrime;
            return(OK);
        case MOS6_SOURCECONDUCT:
            value->rValue = here->MOS6sourceConductance;
            return(OK);
        case MOS6_DRAINCONDUCT:
            value->rValue = here->MOS6drainConductance;
            return(OK);
        case MOS6_SOURCERESIST:
	    if (here->MOS6sNodePrime != here->MOS6sNode)
		value->rValue = 1.0 / here->MOS6sourceConductance;
	    else
		value->rValue = 0.0;
            return(OK);
        case MOS6_DRAINRESIST:
	    if (here->MOS6dNodePrime != here->MOS6dNode)
		value->rValue = 1.0 / here->MOS6drainConductance;
	    else
		value->rValue = 0.0;
            return(OK);
        case MOS6_VON:
            value->rValue = here->MOS6von;
            return(OK);
        case MOS6_VDSAT:
            value->rValue = here->MOS6vdsat;
            return(OK);
        case MOS6_SOURCEVCRIT:
            value->rValue = here->MOS6sourceVcrit;
            return(OK);
        case MOS6_DRAINVCRIT:
            value->rValue = here->MOS6drainVcrit;
            return(OK);
        case MOS6_CD:
            value->rValue = here->MOS6cd;
            return(OK);
        case MOS6_CBS:
            value->rValue = here->MOS6cbs;
            return(OK);
        case MOS6_CBD:
            value->rValue = here->MOS6cbd;
            return(OK);
        case MOS6_GMBS:
            value->rValue = here->MOS6gmbs;
            return(OK);
        case MOS6_GM:
            value->rValue = here->MOS6gm;
            return(OK);
        case MOS6_GDS:
            value->rValue = here->MOS6gds;
            return(OK);
        case MOS6_GBD:
            value->rValue = here->MOS6gbd;
            return(OK);
        case MOS6_GBS:
            value->rValue = here->MOS6gbs;
            return(OK);
        case MOS6_CAPBD:
            value->rValue = here->MOS6capbd;
            return(OK);
        case MOS6_CAPBS:
            value->rValue = here->MOS6capbs;
            return(OK);
        case MOS6_CAPZEROBIASBD:
            value->rValue = here->MOS6Cbd;
            return(OK);
        case MOS6_CAPZEROBIASBDSW:
            value->rValue = here->MOS6Cbdsw;
            return(OK);
        case MOS6_CAPZEROBIASBS:
            value->rValue = here->MOS6Cbs;
            return(OK);
        case MOS6_CAPZEROBIASBSSW:
            value->rValue = here->MOS6Cbssw;
            return(OK);
        case MOS6_VBD:
            value->rValue = *(ckt->CKTstate0 + here->MOS6vbd);
            return(OK);
        case MOS6_VBS:
            value->rValue = *(ckt->CKTstate0 + here->MOS6vbs);
            return(OK);
        case MOS6_VGS:
            value->rValue = *(ckt->CKTstate0 + here->MOS6vgs);
            return(OK);
        case MOS6_VDS:
            value->rValue = *(ckt->CKTstate0 + here->MOS6vds);
            return(OK);
        case MOS6_CAPGS:
            value->rValue = *(ckt->CKTstate0 + here->MOS6capgs);
            return(OK);
        case MOS6_QGS:
            value->rValue = *(ckt->CKTstate0 + here->MOS6qgs);
            return(OK);
        case MOS6_CQGS:
            value->rValue = *(ckt->CKTstate0 + here->MOS6cqgs);
            return(OK);
        case MOS6_CAPGD:
            value->rValue = *(ckt->CKTstate0 + here->MOS6capgd);
            return(OK);
        case MOS6_QGD:
            value->rValue = *(ckt->CKTstate0 + here->MOS6qgd);
            return(OK);
        case MOS6_CQGD:
            value->rValue = *(ckt->CKTstate0 + here->MOS6cqgd);
            return(OK);
        case MOS6_CAPGB:
            value->rValue = *(ckt->CKTstate0 + here->MOS6capgb);
            return(OK);
        case MOS6_QGB:
            value->rValue = *(ckt->CKTstate0 + here->MOS6qgb);
            return(OK);
        case MOS6_CQGB:
            value->rValue = *(ckt->CKTstate0 + here->MOS6cqgb);
            return(OK);
        case MOS6_QBD:
            value->rValue = *(ckt->CKTstate0 + here->MOS6qbd);
            return(OK);
        case MOS6_CQBD:
            value->rValue = *(ckt->CKTstate0 + here->MOS6cqbd);
            return(OK);
        case MOS6_QBS:
            value->rValue = *(ckt->CKTstate0 + here->MOS6qbs);
            return(OK);
        case MOS6_CQBS:
            value->rValue = *(ckt->CKTstate0 + here->MOS6cqbs);
            return(OK);
        case MOS6_L_SENS_DC:
            if(ckt->CKTsenInfo && here->MOS6sens_l){
               value->rValue = *(ckt->CKTsenInfo->SEN_Sap[select->iValue + 1]+
                       here->MOS6senParmNo);
            }
            return(OK);
        case MOS6_L_SENS_REAL:
            if(ckt->CKTsenInfo && here->MOS6sens_l){
               value->rValue = *(ckt->CKTsenInfo->SEN_RHS[select->iValue + 1]+
                       here->MOS6senParmNo);
            }
            return(OK);
        case MOS6_L_SENS_IMAG:
            if(ckt->CKTsenInfo && here->MOS6sens_l){
               value->rValue = *(ckt->CKTsenInfo->SEN_iRHS[select->iValue + 1]+
                       here->MOS6senParmNo);
            }
            return(OK);
        case MOS6_L_SENS_MAG:
            if(ckt->CKTsenInfo && here->MOS6sens_l){
                vr = *(ckt->CKTrhsOld + select->iValue + 1); 
                vi = *(ckt->CKTirhsOld + select->iValue + 1); 
                vm = sqrt(vr*vr + vi*vi);
                if(vm == 0){
                    value->rValue = 0;
                    return(OK);
                }
                sr = *(ckt->CKTsenInfo->SEN_RHS[select->iValue + 1]+
                        here->MOS6senParmNo);
                si = *(ckt->CKTsenInfo->SEN_iRHS[select->iValue + 1]+
                        here->MOS6senParmNo);
                value->rValue = (vr * sr + vi * si)/vm;
            }
            return(OK);
        case MOS6_L_SENS_PH:
            if(ckt->CKTsenInfo && here->MOS6sens_l){
                vr = *(ckt->CKTrhsOld + select->iValue + 1); 
                vi = *(ckt->CKTirhsOld + select->iValue + 1); 
                vm = vr*vr + vi*vi;
                if(vm == 0){
                    value->rValue = 0;
                    return(OK);
                }
                sr = *(ckt->CKTsenInfo->SEN_RHS[select->iValue + 1]+
                        here->MOS6senParmNo);
                si = *(ckt->CKTsenInfo->SEN_iRHS[select->iValue + 1]+
                        here->MOS6senParmNo);
                value->rValue =  (vr * si - vi * sr)/vm;
            }
            return(OK);
        case MOS6_L_SENS_CPLX:
            if(ckt->CKTsenInfo && here->MOS6sens_l){
                value->cValue.real= *(ckt->CKTsenInfo->SEN_RHS[select->iValue + 1]+
                        here->MOS6senParmNo);
                value->cValue.imag= *(ckt->CKTsenInfo->SEN_iRHS[select->iValue + 1]+
                        here->MOS6senParmNo);
            }
            return(OK);
        case MOS6_W_SENS_DC:
            if(ckt->CKTsenInfo && here->MOS6sens_w){
                value->rValue = *(ckt->CKTsenInfo->SEN_Sap[select->iValue + 1]+
                        here->MOS6senParmNo + here->MOS6sens_l);
            }
            return(OK);
        case MOS6_W_SENS_REAL:
            if(ckt->CKTsenInfo && here->MOS6sens_w){
                value->rValue = *(ckt->CKTsenInfo->SEN_RHS[select->iValue + 1]+
                        here->MOS6senParmNo + here->MOS6sens_l);
            }
             return(OK);
        case MOS6_W_SENS_IMAG:
            if(ckt->CKTsenInfo && here->MOS6sens_w){
                value->rValue = *(ckt->CKTsenInfo->SEN_iRHS[select->iValue + 1]+
                        here->MOS6senParmNo + here->MOS6sens_l);
            }
            return(OK);
        case MOS6_W_SENS_MAG:
            if(ckt->CKTsenInfo && here->MOS6sens_w){
                vr = *(ckt->CKTrhsOld + select->iValue + 1); 
                vi = *(ckt->CKTirhsOld + select->iValue + 1); 
                vm = sqrt(vr*vr + vi*vi);
                if(vm == 0){
                    value->rValue = 0;
                    return(OK);
                }
                sr = *(ckt->CKTsenInfo->SEN_RHS[select->iValue + 1]+
                        here->MOS6senParmNo + here->MOS6sens_l);
                si = *(ckt->CKTsenInfo->SEN_iRHS[select->iValue + 1]+
                        here->MOS6senParmNo + here->MOS6sens_l);
                value->rValue = (vr * sr + vi * si)/vm;
            }
            return(OK);
        case MOS6_W_SENS_PH:
            if(ckt->CKTsenInfo && here->MOS6sens_w){
                vr = *(ckt->CKTrhsOld + select->iValue + 1); 
                vi = *(ckt->CKTirhsOld + select->iValue + 1);     
                vm = vr*vr + vi*vi;
                if(vm == 0){
                    value->rValue = 0;
                    return(OK);
                }
                sr = *(ckt->CKTsenInfo->SEN_RHS[select->iValue + 1]+
                        here->MOS6senParmNo + here->MOS6sens_l);
                si = *(ckt->CKTsenInfo->SEN_iRHS[select->iValue + 1]+
                        here->MOS6senParmNo + here->MOS6sens_l);
                value->rValue =  (vr * si - vi * sr)/vm;
            }
                    return(OK);
        case MOS6_W_SENS_CPLX:
            if(ckt->CKTsenInfo && here->MOS6sens_w){
                value->cValue.real= *(ckt->CKTsenInfo->SEN_RHS[select->iValue + 1]+
                        here->MOS6senParmNo + here->MOS6sens_l);
                value->cValue.imag= *(ckt->CKTsenInfo->SEN_iRHS[select->iValue + 1]+
                        here->MOS6senParmNo + here->MOS6sens_l);
            }
            return(OK);
        case MOS6_CB :
            if (ckt->CKTcurrentAnalysis & DOING_AC) {
                errMsg = MALLOC(strlen(msg)+1);
                errRtn = "MOS6ask.c";
                strcpy(errMsg,msg);
                return(E_ASKCURRENT);
            } else {
                value->rValue = here->MOS6cbd + here->MOS6cbs - *(ckt->CKTstate0
                        + here->MOS6cqgb);
            }
            return(OK);
        case MOS6_CG :
            if (ckt->CKTcurrentAnalysis & DOING_AC) {
                errMsg = MALLOC(strlen(msg)+1);
                errRtn = "MOS6ask.c";
                strcpy(errMsg,msg);
                return(E_ASKCURRENT);
            } else if (ckt->CKTcurrentAnalysis & (DOING_DCOP | DOING_TRCV)) {
                value->rValue = 0;
            } else if ((ckt->CKTcurrentAnalysis & DOING_TRAN) && 
                    (ckt->CKTmode & MODETRANOP)) {
                value->rValue = 0;
            } else {
                value->rValue = *(ckt->CKTstate0 + here->MOS6cqgb) +
                        *(ckt->CKTstate0 + here->MOS6cqgd) + *(ckt->CKTstate0 + 
                        here->MOS6cqgs);
            }
            return(OK);
        case MOS6_CS :
            if (ckt->CKTcurrentAnalysis & DOING_AC) {
                errMsg = MALLOC(strlen(msg)+1);
                errRtn = "MOS6ask.c";
                strcpy(errMsg,msg);
                return(E_ASKCURRENT);
            } else {
                value->rValue = -here->MOS6cd;
                value->rValue -= here->MOS6cbd + here->MOS6cbs -
                        *(ckt->CKTstate0 + here->MOS6cqgb);
                if ((ckt->CKTcurrentAnalysis & DOING_TRAN) && 
                        !(ckt->CKTmode & MODETRANOP)) {
                    value->rValue -= *(ckt->CKTstate0 + here->MOS6cqgb) + 
                            *(ckt->CKTstate0 + here->MOS6cqgd) +
                            *(ckt->CKTstate0 + here->MOS6cqgs);
                }
            }
            return(OK);
        case MOS6_POWER :
            if (ckt->CKTcurrentAnalysis & DOING_AC) {
                errMsg = MALLOC(strlen(msg)+1);
                errRtn = "MOS6ask.c";
                strcpy(errMsg,msg);
                return(E_ASKPOWER);
            } else {
                double temp;

                value->rValue = here->MOS6cd * 
                        *(ckt->CKTrhsOld + here->MOS6dNode);
                value->rValue += (here->MOS6cbd + here->MOS6cbs -
                        *(ckt->CKTstate0 + here->MOS6cqgb)) *
                        *(ckt->CKTrhsOld + here->MOS6bNode);
                if ((ckt->CKTcurrentAnalysis & DOING_TRAN) && 
                        !(ckt->CKTmode & MODETRANOP)) {
                    value->rValue += (*(ckt->CKTstate0 + here->MOS6cqgb) + 
                            *(ckt->CKTstate0 + here->MOS6cqgd) +
                            *(ckt->CKTstate0 + here->MOS6cqgs)) *
                            *(ckt->CKTrhsOld + here->MOS6gNode);
                }
                temp = -here->MOS6cd;
                temp -= here->MOS6cbd + here->MOS6cbs ;
                if ((ckt->CKTcurrentAnalysis & DOING_TRAN) && 
                        !(ckt->CKTmode & MODETRANOP)) {
                    temp -= *(ckt->CKTstate0 + here->MOS6cqgb) + 
                            *(ckt->CKTstate0 + here->MOS6cqgd) + 
                            *(ckt->CKTstate0 + here->MOS6cqgs);
                }
                value->rValue += temp * *(ckt->CKTrhsOld + here->MOS6sNode);
            }
            return(OK);
        default:
            return(E_BADPARM);
    }
    /* NOTREACHED */
}

