/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Mathew Lew and Thomas L. Quarles
**********/

/*
 * This routine gives access to the internal device 
 * parameters for BJTs
 */

#include "ngspice.h"
#include "const.h"
#include <stdio.h>
#include "cktdefs.h"
#include "bjtdefs.h"
#include "ifsim.h"
#include "sperror.h"
#include "suffix.h"

/*ARGSUSED*/
int
BJTask(ckt,instPtr,which,value,select)
    CKTcircuit *ckt;
    GENinstance *instPtr;
    int which;
    IFvalue *value;
    IFvalue *select;
{
    BJTinstance *here = (BJTinstance*)instPtr;
    double tmp;
    int itmp;
    double vr;
    double vi;
    double sr;
    double si;
    double vm;
    static char *msg = "Current and power not available for ac analysis";
    switch(which) {
        case BJT_QUEST_FT:
            tmp = MAX(*(ckt->CKTstate0 + here->BJTcqbc),
                *(ckt->CKTstate0 + here->BJTcqbx));
            value->rValue = here->BJTgm/(2 * M_PI *
                MAX(*(ckt->CKTstate0 + here->BJTcqbe),tmp));
            return(OK);
        case BJT_TEMP:
            value->rValue = here->BJTtemp - CONSTCtoK;
            return(OK);
        case BJT_AREA:
            value->rValue = here->BJTarea;
            return(OK);
        case BJT_OFF:
            value->iValue = here->BJToff;
            return(OK);
        case BJT_IC_VBE:
            value->rValue = here->BJTicVBE;
            return(OK);
        case BJT_IC_VCE:
            value->rValue = here->BJTicVCE;
            return(OK);
        case BJT_QUEST_COLNODE:
            value->iValue = here->BJTcolNode;
            return(OK);
        case BJT_QUEST_BASENODE:
            value->iValue = here->BJTbaseNode;
            return(OK);
        case BJT_QUEST_EMITNODE:
            value->iValue = here->BJTemitNode;
            return(OK);
        case BJT_QUEST_SUBSTNODE:
            value->iValue = here->BJTsubstNode;
            return(OK);
        case BJT_QUEST_COLPRIMENODE:
            value->iValue = here->BJTcolPrimeNode;
            return(OK);
        case BJT_QUEST_BASEPRIMENODE:
            value->iValue = here->BJTbasePrimeNode;
            return(OK);
        case BJT_QUEST_EMITPRIMENODE:
            value->iValue = here->BJTemitPrimeNode;
            return(OK);
        case BJT_QUEST_VBE:
            value->rValue = *(ckt->CKTstate0 + here->BJTvbe);
            return(OK);
        case BJT_QUEST_VBC:
            value->rValue = *(ckt->CKTstate0 + here->BJTvbc);
            return(OK);
        case BJT_QUEST_CC:
            value->rValue = *(ckt->CKTstate0 + here->BJTcc);
            return(OK);
        case BJT_QUEST_CB:
            value->rValue = *(ckt->CKTstate0 + here->BJTcb);
            return(OK);
        case BJT_QUEST_GPI:
            value->rValue = *(ckt->CKTstate0 + here->BJTgpi);
            return(OK);
        case BJT_QUEST_GMU:
            value->rValue = *(ckt->CKTstate0 + here->BJTgmu);
            return(OK);
        case BJT_QUEST_GM:
            value->rValue = *(ckt->CKTstate0 + here->BJTgm);
            return(OK);
        case BJT_QUEST_GO:
            value->rValue = *(ckt->CKTstate0 + here->BJTgo);
            return(OK);
        case BJT_QUEST_QBE:
            value->rValue = *(ckt->CKTstate0 + here->BJTqbe);
            return(OK);
        case BJT_QUEST_CQBE:
            value->rValue = *(ckt->CKTstate0 + here->BJTcqbe);
            return(OK);
        case BJT_QUEST_QBC:
            value->rValue = *(ckt->CKTstate0 + here->BJTqbc);
            return(OK);
        case BJT_QUEST_CQBC:
            value->rValue = *(ckt->CKTstate0 + here->BJTcqbc);
            return(OK);
        case BJT_QUEST_QCS:
            value->rValue = *(ckt->CKTstate0 + here->BJTqcs);
            return(OK);
        case BJT_QUEST_CQCS:
            value->rValue = *(ckt->CKTstate0 + here->BJTcqcs);
            return(OK);
        case BJT_QUEST_QBX:
            value->rValue = *(ckt->CKTstate0 + here->BJTqbx);
            return(OK);
        case BJT_QUEST_CQBX:
            value->rValue = *(ckt->CKTstate0 + here->BJTcqbx);
            return(OK);
        case BJT_QUEST_GX:
            value->rValue = *(ckt->CKTstate0 + here->BJTgx);
            return(OK);
        case BJT_QUEST_CEXBC:
            value->rValue = *(ckt->CKTstate0 + here->BJTcexbc);
            return(OK);
        case BJT_QUEST_GEQCB:
            value->rValue = *(ckt->CKTstate0 + here->BJTgeqcb);
            return(OK);
        case BJT_QUEST_GCCS:
            value->rValue = *(ckt->CKTstate0 + here->BJTgccs);
            return(OK);
        case BJT_QUEST_GEQBX:
            value->rValue = *(ckt->CKTstate0 + here->BJTgeqbx);
            return(OK);
    case BJT_QUEST_SENS_DC:
        if(ckt->CKTsenInfo){
           value->rValue = *(ckt->CKTsenInfo->SEN_Sap[select->iValue + 1]+
               here->BJTsenParmNo);
            }
        return(OK);
    case BJT_QUEST_SENS_REAL:
        if(ckt->CKTsenInfo){
           value->rValue = *(ckt->CKTsenInfo->SEN_RHS[select->iValue + 1]+
               here->BJTsenParmNo);
            }
        return(OK);
    case BJT_QUEST_SENS_IMAG:
        if(ckt->CKTsenInfo){
           value->rValue = *(ckt->CKTsenInfo->SEN_iRHS[select->iValue + 1]+
               here->BJTsenParmNo);
            }
        return(OK);
    case BJT_QUEST_SENS_MAG:
        if(ckt->CKTsenInfo){
           vr = *(ckt->CKTrhsOld + select->iValue + 1); 
           vi = *(ckt->CKTirhsOld + select->iValue + 1); 
           vm = sqrt(vr*vr + vi*vi);
           if(vm == 0){
             value->rValue = 0;
             return(OK);
           }
           sr = *(ckt->CKTsenInfo->SEN_RHS[select->iValue + 1]+
                here->BJTsenParmNo);
           si = *(ckt->CKTsenInfo->SEN_iRHS[select->iValue + 1]+
                here->BJTsenParmNo);
               value->rValue = (vr * sr + vi * si)/vm;
            }
        return(OK);
    case BJT_QUEST_SENS_PH:
        if(ckt->CKTsenInfo){
           vr = *(ckt->CKTrhsOld + select->iValue + 1); 
           vi = *(ckt->CKTirhsOld + select->iValue + 1); 
           vm = vr*vr + vi*vi;
           if(vm == 0){
             value->rValue = 0;
             return(OK);
           }
           sr = *(ckt->CKTsenInfo->SEN_RHS[select->iValue + 1]+
                here->BJTsenParmNo);
           si = *(ckt->CKTsenInfo->SEN_iRHS[select->iValue + 1]+
                here->BJTsenParmNo);

               value->rValue =  (vr * si - vi * sr)/vm;
        }
        return(OK);
    case BJT_QUEST_SENS_CPLX:
        if(ckt->CKTsenInfo){
           itmp = select->iValue + 1;
           value->cValue.real= *(ckt->CKTsenInfo->SEN_RHS[itmp]+
               here->BJTsenParmNo);
           value->cValue.imag= *(ckt->CKTsenInfo->SEN_iRHS[itmp]+
               here->BJTsenParmNo);
        }
        return(OK);
        case BJT_QUEST_CS :  
            if (ckt->CKTcurrentAnalysis & DOING_AC) {
                errMsg = MALLOC(strlen(msg)+1);
                errRtn = "BJTask";
                strcpy(errMsg,msg);
                return(E_ASKCURRENT);
            } else if (ckt->CKTcurrentAnalysis & (DOING_DCOP | DOING_TRCV)) {
                value->rValue = 0;
            } else if ((ckt->CKTcurrentAnalysis & DOING_TRAN) && 
                       (ckt->CKTmode & MODETRANOP)) {
                value->rValue = 0;
            } else {
                value->rValue = -*(ckt->CKTstate0 + here->BJTcqcs);
            }
            return(OK);
        case BJT_QUEST_CE :
            if (ckt->CKTcurrentAnalysis & DOING_AC) {
                errMsg = MALLOC(strlen(msg)+1);
                errRtn = "BJTask";
                strcpy(errMsg,msg);
                return(E_ASKCURRENT);
            } else {
                value->rValue = -*(ckt->CKTstate0 + here->BJTcc);
                value->rValue -= *(ckt->CKTstate0 + here->BJTcb);
                if ((ckt->CKTcurrentAnalysis & DOING_TRAN) && 
                        !(ckt->CKTmode & MODETRANOP)) {
                    value->rValue += *(ckt->CKTstate0 + here->BJTcqcs);
                }
            }
            return(OK);
        case BJT_QUEST_POWER :
            if (ckt->CKTcurrentAnalysis & DOING_AC) {
                errMsg = MALLOC(strlen(msg)+1);
                errRtn = "BJTask";
                strcpy(errMsg,msg);
                return(E_ASKPOWER);
            } else {
                value->rValue = *(ckt->CKTstate0 + here->BJTcc) *               
                        *(ckt->CKTrhsOld + here->BJTcolNode);
                value->rValue += *(ckt->CKTstate0 + here->BJTcb) *        
                        *(ckt->CKTrhsOld + here->BJTbaseNode);
                if ((ckt->CKTcurrentAnalysis & DOING_TRAN) && !(ckt->CKTmode & 
                        MODETRANOP)) {
                    value->rValue -= *(ckt->CKTstate0 + here->BJTcqcs) *
                            *(ckt->CKTrhsOld + here->BJTsubstNode);
                }
                tmp = -*(ckt->CKTstate0 + here->BJTcc);
                tmp -= *(ckt->CKTstate0 + here->BJTcb);
                if ((ckt->CKTcurrentAnalysis & DOING_TRAN) && 
                        !(ckt->CKTmode & MODETRANOP)) {
                    tmp += *(ckt->CKTstate0 + here->BJTcqcs);
                }
                value->rValue += tmp * *(ckt->CKTrhsOld + 
                        here->BJTemitNode);
            }
            return(OK);
	case BJT_QUEST_CPI:
	    value->rValue = here->BJTcapbe;
            return(OK);
	case BJT_QUEST_CMU:
	    value->rValue = here->BJTcapbc;
            return(OK);
	case BJT_QUEST_CBX:
	    value->rValue = here->BJTcapbx;
            return(OK);
	case BJT_QUEST_CCS:
	    value->rValue = here->BJTcapcs;
            return(OK);
        default:
            return(E_BADPARM);
    }
    /* NOTREACHED */
}

