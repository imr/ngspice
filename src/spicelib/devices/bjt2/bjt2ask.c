/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Mathew Lew and Thomas L. Quarles
Modified: Alan Gillespie
**********/

/*
 * This routine gives access to the internal device 
 * parameters for BJT2s
 */

#include "ngspice.h"
#include "const.h"
#include "cktdefs.h"
#include "bjt2defs.h"
#include "ifsim.h"
#include "sperror.h"
#include "suffix.h"

/*ARGSUSED*/
int
BJT2ask(CKTcircuit *ckt, GENinstance *instPtr, int which, IFvalue *value,
        IFvalue *select)
{
    BJT2instance *here = (BJT2instance*)instPtr;
    double tmp;
    int itmp;
    double vr;
    double vi;
    double sr;
    double si;
    double vm;
    static char *msg = "Current and power not available for ac analysis";
    switch(which) {
        case BJT2_QUEST_FT:
            tmp = MAX(*(ckt->CKTstate0 + here->BJT2cqbc),
                *(ckt->CKTstate0 + here->BJT2cqbx));
            value->rValue = here->BJT2gm/(2 * M_PI *
                MAX(*(ckt->CKTstate0 + here->BJT2cqbe),tmp));
            return(OK);
        case BJT2_TEMP:
            value->rValue = here->BJT2temp - CONSTCtoK;
            return(OK);
	case BJT2_DTEMP:
            value->rValue = here->BJT2dtemp;
            return(OK);    
        case BJT2_AREA:
            value->rValue = here->BJT2area;
            return(OK);
        case BJT2_AREAB:
            value->rValue = here->BJT2areab;
            return(OK);
        case BJT2_AREAC:
            value->rValue = here->BJT2areac;
            return(OK);      	    
	case BJT2_M:
            value->rValue = here->BJT2m;
            return(OK);    
        case BJT2_OFF:
            value->iValue = here->BJT2off;
            return(OK);
        case BJT2_IC_VBE:
            value->rValue = here->BJT2icVBE;
            return(OK);
        case BJT2_IC_VCE:
            value->rValue = here->BJT2icVCE;
            return(OK);
        case BJT2_QUEST_COLNODE:
            value->iValue = here->BJT2colNode;
            return(OK);
        case BJT2_QUEST_BASENODE:
            value->iValue = here->BJT2baseNode;
            return(OK);
        case BJT2_QUEST_EMITNODE:
            value->iValue = here->BJT2emitNode;
            return(OK);
        case BJT2_QUEST_SUBSTNODE:
            value->iValue = here->BJT2substNode;
            return(OK);
        case BJT2_QUEST_COLPRIMENODE:
            value->iValue = here->BJT2colPrimeNode;
            return(OK);
        case BJT2_QUEST_BASEPRIMENODE:
            value->iValue = here->BJT2basePrimeNode;
            return(OK);
        case BJT2_QUEST_EMITPRIMENODE:
            value->iValue = here->BJT2emitPrimeNode;
            return(OK);
        case BJT2_QUEST_VBE:
            value->rValue = *(ckt->CKTstate0 + here->BJT2vbe);
            return(OK);
        case BJT2_QUEST_VBC:
            value->rValue = *(ckt->CKTstate0 + here->BJT2vbc);
            return(OK);
        case BJT2_QUEST_CC:
            value->rValue = *(ckt->CKTstate0 + here->BJT2cc);
	    value->rValue *= here->BJT2m;
            return(OK);
        case BJT2_QUEST_CB:
            value->rValue = *(ckt->CKTstate0 + here->BJT2cb);
            if (here->BJT2modPtr->BJT2subs==LATERAL) {
                value->rValue -= *(ckt->CKTstate0 + here->BJT2cdsub);
                if ((ckt->CKTcurrentAnalysis & DOING_TRAN) && 
                          !(ckt->CKTmode & MODETRANOP)) {
                      value->rValue -= *(ckt->CKTstate0 + here->BJT2cqsub);
                }
            };
	    value->rValue *= here->BJT2m;
            return(OK);
        case BJT2_QUEST_GPI:
            value->rValue = *(ckt->CKTstate0 + here->BJT2gpi);
	    value->rValue *= here->BJT2m;
            return(OK);
        case BJT2_QUEST_GMU:
            value->rValue = *(ckt->CKTstate0 + here->BJT2gmu);
	    value->rValue *= here->BJT2m;
            return(OK);
        case BJT2_QUEST_GM:
            value->rValue = *(ckt->CKTstate0 + here->BJT2gm);
	    value->rValue *= here->BJT2m;
            return(OK);
        case BJT2_QUEST_GO:
            value->rValue = *(ckt->CKTstate0 + here->BJT2go);
	    value->rValue *= here->BJT2m;
            return(OK);
        case BJT2_QUEST_QBE:
            value->rValue = *(ckt->CKTstate0 + here->BJT2qbe);
	    value->rValue *= here->BJT2m;
            return(OK);
        case BJT2_QUEST_CQBE:
            value->rValue = *(ckt->CKTstate0 + here->BJT2cqbe);
	    value->rValue *= here->BJT2m;
            return(OK);
        case BJT2_QUEST_QBC:
            value->rValue = *(ckt->CKTstate0 + here->BJT2qbc);
	    value->rValue *= here->BJT2m;
            return(OK);
        case BJT2_QUEST_CQBC:
            value->rValue = *(ckt->CKTstate0 + here->BJT2cqbc);
	    value->rValue *= here->BJT2m;
            return(OK);
        case BJT2_QUEST_QSUB:
            value->rValue = *(ckt->CKTstate0 + here->BJT2qsub);
	    value->rValue *= here->BJT2m;
            return(OK);
        case BJT2_QUEST_CQSUB:
            value->rValue = *(ckt->CKTstate0 + here->BJT2cqsub);
	    value->rValue *= here->BJT2m;
            return(OK);
        case BJT2_QUEST_QBX:
            value->rValue = *(ckt->CKTstate0 + here->BJT2qbx);
	    value->rValue *= here->BJT2m;
            return(OK);
        case BJT2_QUEST_CQBX:
            value->rValue = *(ckt->CKTstate0 + here->BJT2cqbx);
	    value->rValue *= here->BJT2m;
            return(OK);
        case BJT2_QUEST_GX:
            value->rValue = *(ckt->CKTstate0 + here->BJT2gx);
	    value->rValue *= here->BJT2m;
            return(OK);
        case BJT2_QUEST_CEXBC:
            value->rValue = *(ckt->CKTstate0 + here->BJT2cexbc);
	    value->rValue *= here->BJT2m;
            return(OK);
        case BJT2_QUEST_GEQCB:
            value->rValue = *(ckt->CKTstate0 + here->BJT2geqcb);
	    value->rValue *= here->BJT2m;
            return(OK);
        case BJT2_QUEST_GCSUB:
            value->rValue = *(ckt->CKTstate0 + here->BJT2gcsub);
	    value->rValue *= here->BJT2m;
            return(OK);
        case BJT2_QUEST_GDSUB:
            value->rValue = *(ckt->CKTstate0 + here->BJT2gdsub);
	    value->rValue *= here->BJT2m;
            return(OK);
        case BJT2_QUEST_GEQBX:
            value->rValue = *(ckt->CKTstate0 + here->BJT2geqbx);
	    value->rValue *= here->BJT2m;
            return(OK);
    case BJT2_QUEST_SENS_DC:
        if(ckt->CKTsenInfo){
           value->rValue = *(ckt->CKTsenInfo->SEN_Sap[select->iValue + 1]+
               here->BJT2senParmNo);
            }
        return(OK);
    case BJT2_QUEST_SENS_REAL:
        if(ckt->CKTsenInfo){
           value->rValue = *(ckt->CKTsenInfo->SEN_RHS[select->iValue + 1]+
               here->BJT2senParmNo);
            }
        return(OK);
    case BJT2_QUEST_SENS_IMAG:
        if(ckt->CKTsenInfo){
           value->rValue = *(ckt->CKTsenInfo->SEN_iRHS[select->iValue + 1]+
               here->BJT2senParmNo);
            }
        return(OK);
    case BJT2_QUEST_SENS_MAG:
        if(ckt->CKTsenInfo){
           vr = *(ckt->CKTrhsOld + select->iValue + 1); 
           vi = *(ckt->CKTirhsOld + select->iValue + 1); 
           vm = sqrt(vr*vr + vi*vi);
           if(vm == 0){
             value->rValue = 0;
             return(OK);
           }
           sr = *(ckt->CKTsenInfo->SEN_RHS[select->iValue + 1]+
                here->BJT2senParmNo);
           si = *(ckt->CKTsenInfo->SEN_iRHS[select->iValue + 1]+
                here->BJT2senParmNo);
               value->rValue = (vr * sr + vi * si)/vm;
            }
        return(OK);
    case BJT2_QUEST_SENS_PH:
        if(ckt->CKTsenInfo){
           vr = *(ckt->CKTrhsOld + select->iValue + 1); 
           vi = *(ckt->CKTirhsOld + select->iValue + 1); 
           vm = vr*vr + vi*vi;
           if(vm == 0){
             value->rValue = 0;
             return(OK);
           }
           sr = *(ckt->CKTsenInfo->SEN_RHS[select->iValue + 1]+
                here->BJT2senParmNo);
           si = *(ckt->CKTsenInfo->SEN_iRHS[select->iValue + 1]+
                here->BJT2senParmNo);

               value->rValue =  (vr * si - vi * sr)/vm;
        }
        return(OK);
    case BJT2_QUEST_SENS_CPLX:
        if(ckt->CKTsenInfo){
           itmp = select->iValue + 1;
           value->cValue.real= *(ckt->CKTsenInfo->SEN_RHS[itmp]+
               here->BJT2senParmNo);
           value->cValue.imag= *(ckt->CKTsenInfo->SEN_iRHS[itmp]+
               here->BJT2senParmNo);
        }
        return(OK);
        case BJT2_QUEST_CS :  
            if (ckt->CKTcurrentAnalysis & DOING_AC) {
                errMsg = MALLOC(strlen(msg)+1);
                errRtn = "BJT2ask";
                strcpy(errMsg,msg);
                return(E_ASKCURRENT);
            } else if (ckt->CKTcurrentAnalysis & (DOING_DCOP | DOING_TRCV)) {
                value->rValue = 0;
            } else if ((ckt->CKTcurrentAnalysis & DOING_TRAN) && 
                       (ckt->CKTmode & MODETRANOP)) {
                value->rValue = 0;
            } else {
                value->rValue = -(here->BJT2modPtr->BJT2subs *
                                   (*(ckt->CKTstate0 + here->BJT2cqsub) +
                                    *(ckt->CKTstate0 + here->BJT2cdsub)));
            }
	    value->rValue *= here->BJT2m;
            return(OK);
        case BJT2_QUEST_CE :
            if (ckt->CKTcurrentAnalysis & DOING_AC) {
                errMsg = MALLOC(strlen(msg)+1);
                errRtn = "BJT2ask";
                strcpy(errMsg,msg);
                return(E_ASKCURRENT);
            } else {
                value->rValue = -*(ckt->CKTstate0 + here->BJT2cc);
                value->rValue -= *(ckt->CKTstate0 + here->BJT2cb);
                if (here->BJT2modPtr->BJT2subs==VERTICAL) {
                  value->rValue += *(ckt->CKTstate0 + here->BJT2cdsub);
                  if ((ckt->CKTcurrentAnalysis & DOING_TRAN) && 
                          !(ckt->CKTmode & MODETRANOP)) {
                           value->rValue += *(ckt->CKTstate0 + here->BJT2cqsub);
                  }
                }
            }
	    value->rValue *= here->BJT2m;
            return(OK);
        case BJT2_QUEST_POWER :
            if (ckt->CKTcurrentAnalysis & DOING_AC) {
                errMsg = MALLOC(strlen(msg)+1);
                errRtn = "BJT2ask";
                strcpy(errMsg,msg);
                return(E_ASKPOWER);
            } else {
                value->rValue = fabs( *(ckt->CKTstate0 + here->BJT2cc) *
                                        (*(ckt->CKTrhsOld + here->BJT2colNode)-
                                         *(ckt->CKTrhsOld + here->BJT2emitNode))
                                    );
                value->rValue +=fabs( *(ckt->CKTstate0 + here->BJT2cb) *
                                        (*(ckt->CKTrhsOld + here->BJT2baseNode)-
                                         *(ckt->CKTrhsOld + here->BJT2emitNode))
                                    );
                value->rValue +=fabs( *(ckt->CKTstate0 + here->BJT2cdsub) *
                                    (*(ckt->CKTrhsOld + here->BJT2substConNode)-
                                     *(ckt->CKTrhsOld + here->BJT2substNode))
                                    );
                if ((ckt->CKTcurrentAnalysis & DOING_TRAN) && !(ckt->CKTmode & 
                        MODETRANOP)) {
                    value->rValue += *(ckt->CKTstate0 + here->BJT2cqsub) *
                            fabs(*(ckt->CKTrhsOld + here->BJT2substConNode)-
                                 *(ckt->CKTrhsOld + here->BJT2substNode));
                }
            }
	    value->rValue *= here->BJT2m;
            return(OK);
	case BJT2_QUEST_CPI:
	    value->rValue = here->BJT2capbe;
	    value->rValue *= here->BJT2m;
            return(OK);
	case BJT2_QUEST_CMU:
	    value->rValue = here->BJT2capbc;
	    value->rValue *= here->BJT2m;
            return(OK);
	case BJT2_QUEST_CBX:
	    value->rValue = here->BJT2capbx;
	    value->rValue *= here->BJT2m;
            return(OK);
	case BJT2_QUEST_CSUB:
	    value->rValue = here->BJT2capsub;
	    value->rValue *= here->BJT2m;
            return(OK);
        default:
            return(E_BADPARM);
    }
    /* NOTREACHED */
}

