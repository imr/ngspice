/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Mathew Lew and Thomas L. Quarles
Model Author: 1990 Michael Schröter TU Dresden
Spice3 Implementation: 2019 Dietmar Warning
**********/

/*
 * This routine gives access to the internal device 
 * parameters for HICUMs
 */

#include "ngspice/ngspice.h"
#include "ngspice/const.h"
#include "ngspice/cktdefs.h"
#include "hicumdefs.h"
#include "ngspice/ifsim.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

/*ARGSUSED*/
int
HICUMask(CKTcircuit *ckt, GENinstance *instPtr, int which, IFvalue *value, IFvalue *select)
{
    HICUMinstance *here = (HICUMinstance*)instPtr;

    NG_IGNORE(select);

    switch(which) {
        case HICUM_AREA:
            value->rValue = here->HICUMarea;
            return(OK);
        case HICUM_OFF:
            value->iValue = here->HICUMoff;
            return(OK);
        case HICUM_IC_VBE:
            value->rValue = here->HICUMicVBE;
            return(OK);
        case HICUM_IC_VCE:
            value->rValue = here->HICUMicVCE;
            return(OK);
        case HICUM_TEMP:
            value->rValue = here->HICUMtemp - CONSTCtoK;
            return(OK);
        case HICUM_M:
            value->rValue = here->HICUMm;
            return(OK);
        case HICUM_QUEST_COLLNODE:
            value->iValue = here->HICUMcollNode;
            return(OK);
        case HICUM_QUEST_BASENODE:
            value->iValue = here->HICUMbaseNode;
            return(OK);
        case HICUM_QUEST_EMITNODE:
            value->iValue = here->HICUMemitNode;
            return(OK);
        case HICUM_QUEST_SUBSNODE:
            value->iValue = here->HICUMsubsNode;
            return(OK);
        case HICUM_QUEST_COLLCINODE:
            value->iValue = here->HICUMcollCINode;
            return(OK);
        case HICUM_QUEST_BASEBPNODE:
            value->iValue = here->HICUMbaseBPNode;
            return(OK);
        case HICUM_QUEST_BASEBINODE:
            value->iValue = here->HICUMbaseBINode;
            return(OK);
        case HICUM_QUEST_EMITEINODE:
            value->iValue = here->HICUMemitEINode;
            return(OK);
        case HICUM_QUEST_SUBSSINODE:
            value->iValue = here->HICUMsubsSINode;
            return(OK);
        case HICUM_QUEST_VBE:
            value->rValue = *(ckt->CKTstate0 + here->HICUMvbiei);
            return(OK);
        case HICUM_QUEST_VBC:
            value->rValue = *(ckt->CKTstate0 + here->HICUMvbici);
            return(OK);
        case HICUM_QUEST_CC:
            value->rValue = *(ckt->CKTstate0 + here->HICUMiciei) -
                            *(ckt->CKTstate0 + here->HICUMibici);
            return(OK);
        case HICUM_QUEST_CB:
            value->rValue = *(ckt->CKTstate0 + here->HICUMibiei) +
                            *(ckt->CKTstate0 + here->HICUMibici) +
                            *(ckt->CKTstate0 + here->HICUMibpci) +
                            *(ckt->CKTstate0 + here->HICUMibpsi);
            return(OK);
        case HICUM_QUEST_CE:
            value->rValue = - *(ckt->CKTstate0 + here->HICUMibiei) -
                            *(ckt->CKTstate0 + here->HICUMibpei) -
                            *(ckt->CKTstate0 + here->HICUMiciei);
            return(OK);
        case HICUM_QUEST_CS:
            value->rValue = *(ckt->CKTstate0 + here->HICUMisici) -
                            *(ckt->CKTstate0 + here->HICUMibpsi);
            return(OK);
        case HICUM_QUEST_GM:
            value->rValue = *(ckt->CKTstate0 + here->HICUMiciei_Vbiei);
            return(OK);
        case HICUM_QUEST_GPI:
            value->rValue = *(ckt->CKTstate0 + here->HICUMibiei_Vbiei);
            return(OK);
        case HICUM_QUEST_GPX:
            value->rValue = *(ckt->CKTstate0 + here->HICUMibpei_Vbpei);
            return(OK);
        case HICUM_QUEST_GMU:
            value->rValue = *(ckt->CKTstate0 + here->HICUMibici_Vbici);
            return(OK);
        case HICUM_QUEST_GX:
            value->rValue = *(ckt->CKTstate0 + here->HICUMibpbi_Vbpbi);
            return(OK);
        case HICUM_QUEST_GO:
            value->rValue = *(ckt->CKTstate0 + here->HICUMiciei_Vbici);
            return(OK);
        case HICUM_QUEST_CJBE:
            value->rValue = here->HICUMcapjei;
            return(OK);
        case HICUM_QUEST_CDBE:
            value->rValue = here->HICUMcapdeix;
            return(OK);
        case HICUM_QUEST_CBEP:
            value->rValue = here->HICUMcapjep;
            return(OK);
        case HICUM_QUEST_CJBC:
            value->rValue = here->HICUMcapjci;
            return(OK);
        case HICUM_QUEST_CBCXI:
            value->rValue = here->HICUMcapjcx_t_i;
            return(OK);
        case HICUM_QUEST_CBCXII:
            value->rValue = here->HICUMcapjcx_t_ii;
            return(OK);
        case HICUM_QUEST_CSCP:
            value->rValue = here->HICUMcapscp;
            return(OK);
        case HICUM_QUEST_QBE:
            value->rValue = *(ckt->CKTstate0 + here->HICUMqjei);
            return(OK);
        case HICUM_QUEST_QBC:
            value->rValue = *(ckt->CKTstate0 + here->HICUMqjci);
            return(OK);
        case HICUM_QUEST_POWER:
            value->rValue = fabs(*(ckt->CKTstate0 + here->HICUMiciei)) * fabs(*(ckt->CKTstate0 + here->HICUMvbiei) - *(ckt->CKTstate0 + here->HICUMvbici)) +
                            fabs(*(ckt->CKTstate0 + here->HICUMibiei) * *(ckt->CKTstate0 + here->HICUMvbiei)) +
                            fabs(*(ckt->CKTstate0 + here->HICUMibpei) * *(ckt->CKTstate0 + here->HICUMvbpei)) +
                            fabs(*(ckt->CKTstate0 + here->HICUMibici) * *(ckt->CKTstate0 + here->HICUMvbici)) +
                            fabs(*(ckt->CKTstate0 + here->HICUMibpci) * *(ckt->CKTstate0 + here->HICUMvbpci)) +
                            fabs(*(ckt->CKTstate0 + here->HICUMibpsi) * *(ckt->CKTstate0 + here->HICUMvsici));
            return(OK);
        default:
            return(E_BADPARM);
    }
    /* NOTREACHED */
}

