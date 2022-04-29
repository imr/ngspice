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

#include "ngspice/ngspice.h"
#include "ngspice/const.h"
#include "ngspice/cktdefs.h"
#include "vbicdefs.h"
#include "ngspice/ifsim.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

/*ARGSUSED*/
int
VBICask(CKTcircuit *ckt, GENinstance *instPtr, int which, IFvalue *value, IFvalue *select)
{
    IFvalue IC, IB, IE, IS;
    VBICinstance *here = (VBICinstance*)instPtr;

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
            value->rValue *= VBICmodPtr(here)->VBICtype;
            return(OK);
        case VBIC_QUEST_CB:
            value->rValue = *(ckt->CKTstate0 + here->VBICibe) +
                            *(ckt->CKTstate0 + here->VBICibc) +
                            *(ckt->CKTstate0 + here->VBICibex) +
                            *(ckt->CKTstate0 + here->VBICibep) +
                            *(ckt->CKTstate0 + here->VBICiccp);
            value->rValue *= VBICmodPtr(here)->VBICtype;
            return(OK);
        case VBIC_QUEST_CE:
            value->rValue = - *(ckt->CKTstate0 + here->VBICibe) -
                            *(ckt->CKTstate0 + here->VBICibex) -
                            *(ckt->CKTstate0 + here->VBICitzf) +
                            *(ckt->CKTstate0 + here->VBICitzr);
            value->rValue *= VBICmodPtr(here)->VBICtype;
            return(OK);
        case VBIC_QUEST_CS:
            value->rValue = *(ckt->CKTstate0 + here->VBICiccp) -
                            *(ckt->CKTstate0 + here->VBICibcp);
            value->rValue *= VBICmodPtr(here)->VBICtype;
            return(OK);
        case VBIC_QUEST_POWER:
            VBICask(ckt, instPtr, VBIC_QUEST_CC, &IC, select);
            VBICask(ckt, instPtr, VBIC_QUEST_CB, &IB, select);
            VBICask(ckt, instPtr, VBIC_QUEST_CE, &IE, select);
            VBICask(ckt, instPtr, VBIC_QUEST_CS, &IS, select);
            value->rValue = fabs(*(ckt->CKTstate0 + here->VBICibe) * *(ckt->CKTstate0 + here->VBICvbei)) +
                            fabs(*(ckt->CKTstate0 + here->VBICibc) * *(ckt->CKTstate0 + here->VBICvbci)) +
                            fabs(*(ckt->CKTstate0 + here->VBICitzf) - *(ckt->CKTstate0 + here->VBICitzr)) 
                                * fabs(*(ckt->CKTstate0 + here->VBICvbei) - *(ckt->CKTstate0 + here->VBICvbci)) +
                            fabs(*(ckt->CKTstate0 + here->VBICibex) * *(ckt->CKTstate0 + here->VBICvbex)) +
                            fabs(*(ckt->CKTstate0 + here->VBICibep) * *(ckt->CKTstate0 + here->VBICvbep)) +
                            fabs(*(ckt->CKTstate0 + here->VBICibcp) * *(ckt->CKTstate0 + here->VBICvbcp)) +
                            fabs(*(ckt->CKTstate0 + here->VBICiccp)) 
                                * fabs(*(ckt->CKTstate0 + here->VBICvbep) - *(ckt->CKTstate0 + here->VBICvbcp)) +
                            fabs(IC.rValue * IC.rValue * here->VBICtextCollResist) +
                            fabs(IC.rValue * *(ckt->CKTstate0 + here->VBICvrci)) +
                            fabs(IB.rValue * IB.rValue * here->VBICtextBaseResist) +
                            fabs(IB.rValue * *(ckt->CKTstate0 + here->VBICvrbi)) +
                            fabs(IE.rValue * IE.rValue * here->VBICtemitterResist) +
                            fabs(IS.rValue * *(ckt->CKTstate0 + here->VBICvrbp));
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
        default:
            return(E_BADPARM);
    }
    /* NOTREACHED */
}

