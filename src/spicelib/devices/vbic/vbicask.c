/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Mathew Lew and Thomas L. Quarles
Model Author: 1995 Colin McAndrew Motorola
Spice3 Implementation: 2003 Dietmar Warning DAnalyse GmbH
**********/

/*
 * This routine gives access to the internal device 
 * parameters for VBICs
 */

#include "ngspice.h"
#include "const.h"
#include "cktdefs.h"
#include "vbicdefs.h"
#include "ifsim.h"
#include "sperror.h"
#include "suffix.h"

/*ARGSUSED*/
int
VBICask(CKTcircuit *ckt, GENinstance *instPtr, int which, IFvalue *value, IFvalue *select)
{
    VBICinstance *here = (VBICinstance*)instPtr;
    int itmp;
    double vr;
    double vi;
    double sr;
    double si;
    double vm;

    switch(which) {
        case VBIC_AREA:
            value->rValue = here->VBICarea;
            return(OK);
        case VBIC_OFF:
            value->iValue = here->VBICoff;
            return(OK);
        case VBIC_IC_VBE:
            value->rValue = here->VBICicVBE;
            return(OK);
        case VBIC_IC_VCE:
            value->rValue = here->VBICicVCE;
            return(OK);
        case VBIC_TEMP:
            value->rValue = here->VBICtemp - CONSTCtoK;
            return(OK);
        case VBIC_M:
            value->rValue = here->VBICm;
            return(OK);
        case VBIC_QUEST_COLLNODE:
            value->iValue = here->VBICcollNode;
            return(OK);
        case VBIC_QUEST_BASENODE:
            value->iValue = here->VBICbaseNode;
            return(OK);
        case VBIC_QUEST_EMITNODE:
            value->iValue = here->VBICemitNode;
            return(OK);
        case VBIC_QUEST_SUBSNODE:
            value->iValue = here->VBICsubsNode;
            return(OK);
        case VBIC_QUEST_COLLCXNODE:
            value->iValue = here->VBICcollCXNode;
            return(OK);
        case VBIC_QUEST_BASEBXNODE:
            value->iValue = here->VBICbaseBXNode;
            return(OK);
        case VBIC_QUEST_EMITEINODE:
            value->iValue = here->VBICemitEINode;
            return(OK);
        case VBIC_QUEST_SUBSSINODE:
            value->iValue = here->VBICsubsSINode;
            return(OK);
        case VBIC_QUEST_VBE:
            value->rValue = *(ckt->CKTstate0 + here->VBICvbei);
            return(OK);
        case VBIC_QUEST_VBC:
            value->rValue = *(ckt->CKTstate0 + here->VBICvbci);
            return(OK);
        case VBIC_QUEST_CC:
            value->rValue = *(ckt->CKTstate0 + here->VBICitzf) -
                            *(ckt->CKTstate0 + here->VBICitzr) -
                            *(ckt->CKTstate0 + here->VBICibc);
            return(OK);
        case VBIC_QUEST_CB:
            value->rValue = *(ckt->CKTstate0 + here->VBICibe) +
                            *(ckt->CKTstate0 + here->VBICibc) +
                            *(ckt->CKTstate0 + here->VBICibex) +
                            *(ckt->CKTstate0 + here->VBICibep) +
                            *(ckt->CKTstate0 + here->VBICiccp);
            return(OK);
        case VBIC_QUEST_CE:
            value->rValue = - *(ckt->CKTstate0 + here->VBICibe) -
                            *(ckt->CKTstate0 + here->VBICibex) -
                            *(ckt->CKTstate0 + here->VBICitzf) +
                            *(ckt->CKTstate0 + here->VBICitzr);
            return(OK);
        case VBIC_QUEST_CS:
            value->rValue = *(ckt->CKTstate0 + here->VBICiccp) -
                            *(ckt->CKTstate0 + here->VBICibcp);
            return(OK);
        case VBIC_QUEST_POWER:
            value->rValue = fabs(*(ckt->CKTstate0 + here->VBICitzf) - *(ckt->CKTstate0 + here->VBICitzr)) 
                            * fabs(*(ckt->CKTstate0 + here->VBICvbei) - *(ckt->CKTstate0 + here->VBICvbci)) +
                            fabs(*(ckt->CKTstate0 + here->VBICibe) * *(ckt->CKTstate0 + here->VBICvbei)) +
                            fabs(*(ckt->CKTstate0 + here->VBICibex) * *(ckt->CKTstate0 + here->VBICvbex)) +
                            fabs(*(ckt->CKTstate0 + here->VBICibc) * *(ckt->CKTstate0 + here->VBICvbci)) +
                            fabs(*(ckt->CKTstate0 + here->VBICibcp) * *(ckt->CKTstate0 + here->VBICvbcp)) +
                            fabs(*(ckt->CKTstate0 + here->VBICiccp)) 
                            * fabs(*(ckt->CKTstate0 + here->VBICvbep) - *(ckt->CKTstate0 + here->VBICvbcp));
            return(OK);
        case VBIC_QUEST_GM:
            value->rValue = *(ckt->CKTstate0 + here->VBICitzf_Vbei);
            return(OK);
        case VBIC_QUEST_GO:
            value->rValue = *(ckt->CKTstate0 + here->VBICitzf_Vbci);
            return(OK);
        case VBIC_QUEST_GPI:
            value->rValue = *(ckt->CKTstate0 + here->VBICibe_Vbei);
            return(OK);
        case VBIC_QUEST_GMU:
            value->rValue = *(ckt->CKTstate0 + here->VBICibc_Vbci);
            return(OK);
        case VBIC_QUEST_GX:
            value->rValue = *(ckt->CKTstate0 + here->VBICirbi_Vrbi);
            return(OK);
        case VBIC_QUEST_CBE:
            value->rValue = here->VBICcapbe;
            return(OK);
        case VBIC_QUEST_CBEX:
            value->rValue = here->VBICcapbex;
            return(OK);
        case VBIC_QUEST_CBC:
            value->rValue = here->VBICcapbc;
            return(OK);
        case VBIC_QUEST_CBCX:
            value->rValue = here->VBICcapbcx;
            return(OK);
        case VBIC_QUEST_CBEP:
            value->rValue = here->VBICcapbep;
            return(OK);
        case VBIC_QUEST_CBCP:
            value->rValue = here->VBICcapbcp;
            return(OK);
        case VBIC_QUEST_QBE:
            value->rValue = *(ckt->CKTstate0 + here->VBICqbe);
            return(OK);
        case VBIC_QUEST_QBC:
            value->rValue = *(ckt->CKTstate0 + here->VBICqbc);
            return(OK);
        case VBIC_QUEST_SENS_DC:
            if(ckt->CKTsenInfo){
               value->rValue = *(ckt->CKTsenInfo->SEN_Sap[select->iValue + 1]+
                   here->VBICsenParmNo);
            }
            return(OK);
        case VBIC_QUEST_SENS_REAL:
            if(ckt->CKTsenInfo){
               value->rValue = *(ckt->CKTsenInfo->SEN_RHS[select->iValue + 1]+
                   here->VBICsenParmNo);
            }
            return(OK);
        case VBIC_QUEST_SENS_IMAG:
            if(ckt->CKTsenInfo){
               value->rValue = *(ckt->CKTsenInfo->SEN_iRHS[select->iValue + 1]+
                   here->VBICsenParmNo);
            }
            return(OK);
        case VBIC_QUEST_SENS_MAG:
            if(ckt->CKTsenInfo){
               vr = *(ckt->CKTrhsOld + select->iValue + 1); 
               vi = *(ckt->CKTirhsOld + select->iValue + 1); 
               vm = sqrt(vr*vr + vi*vi);
               if(vm == 0){
                 value->rValue = 0;
                 return(OK);
               }
               sr = *(ckt->CKTsenInfo->SEN_RHS[select->iValue + 1]+
                    here->VBICsenParmNo);
               si = *(ckt->CKTsenInfo->SEN_iRHS[select->iValue + 1]+
                    here->VBICsenParmNo);
                   value->rValue = (vr * sr + vi * si)/vm;
            }
            return(OK);
        case VBIC_QUEST_SENS_PH:
            if(ckt->CKTsenInfo){
               vr = *(ckt->CKTrhsOld + select->iValue + 1); 
               vi = *(ckt->CKTirhsOld + select->iValue + 1); 
               vm = vr*vr + vi*vi;
               if(vm == 0){
                 value->rValue = 0;
                 return(OK);
               }
               sr = *(ckt->CKTsenInfo->SEN_RHS[select->iValue + 1]+
                    here->VBICsenParmNo);
               si = *(ckt->CKTsenInfo->SEN_iRHS[select->iValue + 1]+
                    here->VBICsenParmNo);
       
                   value->rValue =  (vr * si - vi * sr)/vm;
            }
            return(OK);
        case VBIC_QUEST_SENS_CPLX:
            if(ckt->CKTsenInfo){
               itmp = select->iValue + 1;
               value->cValue.real= *(ckt->CKTsenInfo->SEN_RHS[itmp]+
                   here->VBICsenParmNo);
               value->cValue.imag= *(ckt->CKTsenInfo->SEN_iRHS[itmp]+
                   here->VBICsenParmNo);
            }
            return(OK);
        default:
            return(E_BADPARM);
    }
    /* NOTREACHED */
}

