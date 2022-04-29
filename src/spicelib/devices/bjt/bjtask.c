/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Mathew Lew and Thomas L. Quarles
**********/

/*
 * This routine gives access to the internal device 
 * parameters for BJTs
 */

#include "ngspice/ngspice.h"
#include "ngspice/const.h"
#include "ngspice/cktdefs.h"
#include "bjtdefs.h"
#include "ngspice/ifsim.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

/*ARGSUSED*/
int
BJTask(CKTcircuit *ckt, GENinstance *instPtr, int which, IFvalue *value, IFvalue *select)
{
    BJTinstance *here = (BJTinstance*)instPtr;
    double tmp;
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
        case BJT_DTEMP:
            value->rValue = here->BJTdtemp;
            return(OK);            
        case BJT_AREA:
            value->rValue = here->BJTarea;
            return(OK);
        case BJT_AREAB:
            value->rValue = here->BJTareab;
            return(OK);
        case BJT_AREAC:
            value->rValue = here->BJTareac;
            return(OK);        
        case BJT_M:
            value->rValue = here->BJTm;
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
        case BJT_QUEST_COLLCXNODE:
            value->iValue = here->BJTcollCXNode;
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
            value->rValue *= here->BJTm * BJTmodPtr(here)->BJTtype;
            return(OK);
        case BJT_QUEST_CB:
            value->rValue = *(ckt->CKTstate0 + here->BJTcb);
            if (BJTmodPtr(here)->BJTsubs==LATERAL) {
                value->rValue -= *(ckt->CKTstate0 + here->BJTcdsub);
                if ((ckt->CKTcurrentAnalysis & DOING_TRAN) && 
                          !(ckt->CKTmode & MODETRANOP)) {
                      value->rValue -= *(ckt->CKTstate0 + here->BJTcqsub);
                }
            }
            value->rValue *= here->BJTm * BJTmodPtr(here)->BJTtype;
            return(OK);
        case BJT_QUEST_GPI:
            value->rValue = *(ckt->CKTstate0 + here->BJTgpi);
            value->rValue *= here->BJTm;
            return(OK);
        case BJT_QUEST_GMU:
            value->rValue = *(ckt->CKTstate0 + here->BJTgmu);
            value->rValue *= here->BJTm;
            return(OK);
        case BJT_QUEST_GM:
            value->rValue = *(ckt->CKTstate0 + here->BJTgm);
            value->rValue *= here->BJTm;
            return(OK);
        case BJT_QUEST_GO:
            value->rValue = *(ckt->CKTstate0 + here->BJTgo);
            value->rValue *= here->BJTm;
            return(OK);
        case BJT_QUEST_QBE:
            value->rValue = *(ckt->CKTstate0 + here->BJTqbe);
            value->rValue *= here->BJTm;
            return(OK);
        case BJT_QUEST_CQBE:
            value->rValue = *(ckt->CKTstate0 + here->BJTcqbe);
            value->rValue *= here->BJTm;
            return(OK);
        case BJT_QUEST_QBC:
            value->rValue = *(ckt->CKTstate0 + here->BJTqbc);
            value->rValue *= here->BJTm;
            return(OK);
        case BJT_QUEST_CQBC:
            value->rValue = *(ckt->CKTstate0 + here->BJTcqbc);
            value->rValue *= here->BJTm;
            return(OK);
        case BJT_QUEST_QSUB:
            value->rValue = *(ckt->CKTstate0 + here->BJTqsub);
            value->rValue *= here->BJTm;
            return(OK);
        case BJT_QUEST_CQSUB:
            value->rValue = *(ckt->CKTstate0 + here->BJTcqsub);
            value->rValue *= here->BJTm;
            return(OK);
        case BJT_QUEST_QBX:
            value->rValue = *(ckt->CKTstate0 + here->BJTqbx);
            value->rValue *= here->BJTm;
            return(OK);
        case BJT_QUEST_CQBX:
            value->rValue = *(ckt->CKTstate0 + here->BJTcqbx);
            value->rValue *= here->BJTm;
            return(OK);
        case BJT_QUEST_GX:
            value->rValue = *(ckt->CKTstate0 + here->BJTgx);
            value->rValue *= here->BJTm;
            return(OK);
        case BJT_QUEST_CEXBC:
            value->rValue = *(ckt->CKTstate0 + here->BJTcexbc);
            value->rValue *= here->BJTm;
            return(OK);
        case BJT_QUEST_GEQCB:
            value->rValue = *(ckt->CKTstate0 + here->BJTgeqcb);
            value->rValue *= here->BJTm;
            return(OK);
        case BJT_QUEST_GCSUB:
            value->rValue = *(ckt->CKTstate0 + here->BJTgcsub);
            value->rValue *= here->BJTm;
            return(OK);
        case BJT_QUEST_GDSUB:
            value->rValue = *(ckt->CKTstate0 + here->BJTgdsub);
            value->rValue *= here->BJTm;
            return(OK);
        case BJT_QUEST_GEQBX:
            value->rValue = *(ckt->CKTstate0 + here->BJTgeqbx);
            value->rValue *= here->BJTm;
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
               int itmp;
               itmp = select->iValue + 1;
               value->cValue.real= *(ckt->CKTsenInfo->SEN_RHS[itmp]+
                   here->BJTsenParmNo);
               value->cValue.imag= *(ckt->CKTsenInfo->SEN_iRHS[itmp]+
                   here->BJTsenParmNo);
            }
            return(OK);
        case BJT_QUEST_CS :  
            if (ckt->CKTcurrentAnalysis & DOING_AC) {
                errMsg = TMALLOC(char, strlen(msg) + 1);
                errRtn = "BJTask";
                strcpy(errMsg,msg);
                return(E_ASKCURRENT);
            } else if (ckt->CKTcurrentAnalysis & (DOING_DCOP | DOING_TRCV)) {
                value->rValue = 0;
            } else if ((ckt->CKTcurrentAnalysis & DOING_TRAN) && 
                       (ckt->CKTmode & MODETRANOP)) {
                value->rValue = 0;
            } else {
                value->rValue = -(BJTmodPtr(here)->BJTsubs *
                                   (*(ckt->CKTstate0 + here->BJTcqsub) +
                                    *(ckt->CKTstate0 + here->BJTcdsub)));
            }
            value->rValue *= here->BJTm * BJTmodPtr(here)->BJTtype;
            return(OK);
        case BJT_QUEST_CE :
            if (ckt->CKTcurrentAnalysis & DOING_AC) {
                errMsg = TMALLOC(char, strlen(msg) + 1);
                errRtn = "BJTask";
                strcpy(errMsg,msg);
                return(E_ASKCURRENT);
            } else {
                value->rValue = -*(ckt->CKTstate0 + here->BJTcc);
                value->rValue -= *(ckt->CKTstate0 + here->BJTcb);
                if (BJTmodPtr(here)->BJTsubs==VERTICAL) {
                    value->rValue += *(ckt->CKTstate0 + here->BJTcdsub);
                    if ((ckt->CKTcurrentAnalysis & DOING_TRAN) && 
                            !(ckt->CKTmode & MODETRANOP)) {
                             value->rValue += *(ckt->CKTstate0 + here->BJTcqsub);
                    }
                }
            }
            value->rValue *= here->BJTm * BJTmodPtr(here)->BJTtype;
            return(OK);
        case BJT_QUEST_POWER :
            if (ckt->CKTcurrentAnalysis & DOING_AC) {
                errMsg = TMALLOC(char, strlen(msg) + 1);
                errRtn = "BJTask";
                strcpy(errMsg,msg);
                return(E_ASKPOWER);
            } else {
                value->rValue = fabs( *(ckt->CKTstate0 + here->BJTcc) *
                                        (*(ckt->CKTrhsOld + here->BJTcolNode)-
                                         *(ckt->CKTrhsOld + here->BJTemitNode))
                                    );
                value->rValue +=fabs( *(ckt->CKTstate0 + here->BJTcb) *
                                        (*(ckt->CKTrhsOld + here->BJTbaseNode)-
                                         *(ckt->CKTrhsOld + here->BJTemitNode))
                                    );
                value->rValue +=fabs( *(ckt->CKTstate0 + here->BJTcdsub) *
                                    (*(ckt->CKTrhsOld + here->BJTsubstConNode)-
                                     *(ckt->CKTrhsOld + here->BJTsubstNode))
                                    );
                if ((ckt->CKTcurrentAnalysis & DOING_TRAN) && !(ckt->CKTmode & 
                        MODETRANOP)) {
                    value->rValue += *(ckt->CKTstate0 + here->BJTcqsub) *
                            fabs(*(ckt->CKTrhsOld + here->BJTsubstConNode)-
                                 *(ckt->CKTrhsOld + here->BJTsubstNode));
                }
            }
            value->rValue *= here->BJTm;
            return(OK);
        case BJT_QUEST_CPI:
            value->rValue = here->BJTcapbe;
            value->rValue *= here->BJTm;
            return(OK);
        case BJT_QUEST_CMU:
            value->rValue = here->BJTcapbc;
            value->rValue *= here->BJTm;
            return(OK);
        case BJT_QUEST_CBX:
            value->rValue = here->BJTcapbx;
            value->rValue *= here->BJTm;
            return(OK);
        case BJT_QUEST_CSUB:
            value->rValue = here->BJTcapsub;
            value->rValue *= here->BJTm;
            return(OK);
        default:
            return(E_BADPARM);
    }
    /* NOTREACHED */
}

