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
#include "hicum2defs.h"
#include "ngspice/ifsim.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

/*ARGSUSED*/
int
HICUMask(CKTcircuit *ckt, GENinstance *instPtr, int which, IFvalue *value, IFvalue *select)
{
    HICUMinstance *here = (HICUMinstance*)instPtr;

    NG_IGNORE(select);
    double g_be;
    IFvalue IC, IB, RPIi, RPIx, GMi;
    IFvalue CPIi, CPIx, CMUi, CMUx;
    IFvalue rcx_t, re_t, rb, BETAAC;

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
/* voltages */
        case HICUM_QUEST_VBE:
            value->rValue = *(ckt->CKTstate0 + here->HICUMbaseNode)-*(ckt->CKTstate0 + here->HICUMemitNode);
            return(OK);
        case HICUM_QUEST_VBC:
            value->rValue = *(ckt->CKTstate0 + here->HICUMbaseNode)-*(ckt->CKTstate0 + here->HICUMcollNode);
            return(OK);
        case HICUM_QUEST_VCE:
            value->rValue = *(ckt->CKTstate0 + here->HICUMcollNode)-*(ckt->CKTstate0 + here->HICUMemitNode);
            return(OK);
        case HICUM_QUEST_VSC:
            value->rValue = *(ckt->CKTstate0 + here->HICUMcollNode)-*(ckt->CKTstate0 + here->HICUMsubsNode);
            return(OK);
/* currents */
        case HICUM_QUEST_CC:
            value->rValue = 0.0;
            // value->rValue = *(ckt->CKTstate0 + here->HICUMiciei) - // TODO: why does this not work ???
            //                 *(ckt->CKTstate0 + here->HICUMibici) - // these counter shoud be defined and expand to a integer
            //                 *(ckt->CKTstate0 + here->HICUMibpci) -
            //                 *(ckt->CKTstate0 + here->HICUMisici);
            return(OK);
        case HICUM_QUEST_CB:
            value->rValue = 0.0;
            // value->rValue = *(ckt->CKTstate0 + here->HICUMibiei) +
            //                 *(ckt->CKTstate0 + here->HICUMibici) +
            //                 *(ckt->CKTstate0 + here->HICUMibpci) +
            //                 *(ckt->CKTstate0 + here->HICUMibpsi);
            return(OK);
        case HICUM_QUEST_CE:
            value->rValue = 0.0;
            // value->rValue = - *(ckt->CKTstate0 + here->HICUMibiei) -
            //                 *(ckt->CKTstate0 + here->HICUMibpei) -
            //                 *(ckt->CKTstate0 + here->HICUMiciei);
            return(OK);
        case HICUM_QUEST_CS:
            value->rValue = 0.0;
            // value->rValue = *(ckt->CKTstate0 + here->HICUMisici) -
            //                 *(ckt->CKTstate0 + here->HICUMibpsi);
            return(OK);
        case HICUM_QUEST_CAVL:
            value->rValue = here->HICUMiavl;
            return(OK);
/* resistances */
        case HICUM_QUEST_RCX_T:
            value->rValue = here->HICUMrcx_t;
            return(OK);
        case HICUM_QUEST_RE_T:
            value->rValue = here->HICUMre_t;
            return(OK);
        case HICUM_QUEST_IT:
            value->rValue = *(ckt->CKTstate0 + here->HICUMiciei);
            return(OK);
        case HICUM_QUEST_RBI:
            value->rValue = here->HICUMrbi;
            return(OK);
        case HICUM_QUEST_RB:
            value->rValue = here->HICUMrbi + here->HICUMrbx_t;
            return(OK);
/* transconductances and capacitances */
        case HICUM_QUEST_BETADC:
            // HICUMask(CKTcircuit *ckt, GENinstance *instPtr, int which, IFvalue *value, IFvalue *select)
            HICUMask(ckt, instPtr, HICUM_QUEST_CC, &IC, select);
            HICUMask(ckt, instPtr, HICUM_QUEST_CB, &IB, select);
            if (IB.rValue != 0.0) {
                value->rValue = IC.rValue/IB.rValue;
            } else {
                value->rValue = 0.0;
            }
            return(OK);
        case HICUM_QUEST_GMI:
            value->rValue = 0.0;
            // value->rValue = *(ckt->CKTstate0 + here->HICUMiciei_Vbiei); // TODO: Check sign vs VA-Code in ADS
            return(OK);
        case HICUM_QUEST_GMS:
            value->rValue = 0.0;
            // value->rValue = *(ckt->CKTstate0 + here->HICUMibpsi_Vbpci); // TODO: Check sign vs VA-Code in ADS
            return(OK);
        case HICUM_QUEST_RPII:
            value->rValue = 0.0;
            // value->rValue = 1/( *(ckt->CKTstate0 + here->HICUMibiei_Vbiei) + ckt->CKTgmin); // TODO: Check sign vs VA-Code in ADS
            return(OK);
        case HICUM_QUEST_RPIX:
            value->rValue = 0.0;
            // value->rValue = 1/( *(ckt->CKTstate0 + here->HICUMibpei_Vbpei) + ckt->CKTgmin); // TODO: Check sign vs VA-Code in ADS
            return(OK);
        case HICUM_QUEST_RMUI:
            value->rValue = 0.0;
            // value->rValue = 1/( *(ckt->CKTstate0 + here->HICUMibici_Vbici) + ckt->CKTgmin); // TODO: Check sign vs VA-Code in ADS
            return(OK);
        case HICUM_QUEST_RMUX:
            value->rValue = 0.0;
            // value->rValue = 1/( *(ckt->CKTstate0 + here->HICUMibpci_Vbpci) + ckt->CKTgmin); // TODO: Check sign vs VA-Code in ADS
            return(OK);
        case HICUM_QUEST_ROI:
            value->rValue = 0.0;
            // value->rValue = 1/( *(ckt->CKTstate0 + here->HICUMiciei_Vbiei) + ckt->CKTgmin); // TODO: Check sign vs VA-Code in ADS
            return(OK);
        case HICUM_QUEST_CPII:
            value->rValue = here->HICUMcapjei + here->HICUMcapdeix;
            return(OK);
        case HICUM_QUEST_CPIX:
            value->rValue = here->HICUMcapjep + here->HICUMcbepar;
            return(OK);
        case HICUM_QUEST_CMUI:
            value->rValue = here->HICUMcapjci + here->HICUMcapdci;
            return(OK);
        case HICUM_QUEST_CMUX:
            value->rValue = here->HICUMcapjcx_t_i + here->HICUMcapjcx_t_ii + here->HICUMcbcpar + here->HICUMcapdsu;
            return(OK);
        case HICUM_QUEST_CCS:
            value->rValue = here->HICUMcapjs + here->HICUMcapscp;
            return(OK);
        case HICUM_QUEST_CRBI:
            value->rValue = here->HICUMcaprbi;
            return(OK);
        case HICUM_QUEST_BETAAC:
            // HICUMask(CKTcircuit *ckt, GENinstance *instPtr, int which, IFvalue *value, IFvalue *select)
            HICUMask(ckt, instPtr, HICUM_QUEST_RPII, &RPIi, select);
            HICUMask(ckt, instPtr, HICUM_QUEST_RPIX, &RPIx, select);
            HICUMask(ckt, instPtr, HICUM_QUEST_GMI, &GMi, select);
            g_be = 1/(RPIi.rValue + RPIx.rValue);
            if (g_be > 0.0) {
                value->rValue = GMi.rValue/g_be;
            } else {
                value->rValue = 0.0;
            }
            return(OK);
/* transit time */
        case HICUM_QUEST_TF:
            value->rValue = here->HICUMtf;
            return(OK);
        case HICUM_QUEST_CBEI:
            value->rValue = *(ckt->CKTstate0 + here->HICUMibiei);
            return(OK);
        case HICUM_QUEST_CBCI:
            value->rValue = *(ckt->CKTstate0 + here->HICUMibici);
            return(OK);
        case HICUM_QUEST_FT:
            // FT = GMi/(2*`M_PI*(CPIi+CPIx+CMUi+CMUx+(rcx_t+re_t+(re_t+rb)/BETAAC)*GMi*(CMUi+CMUx)));
            HICUMask(ckt, instPtr, HICUM_QUEST_GMI, &GMi, select);
            HICUMask(ckt, instPtr, HICUM_QUEST_CPII, &CPIi, select);
            HICUMask(ckt, instPtr, HICUM_QUEST_CPIX, &CPIx, select);
            HICUMask(ckt, instPtr, HICUM_QUEST_CMUI, &CMUi, select);
            HICUMask(ckt, instPtr, HICUM_QUEST_CMUX, &CMUx, select);
            HICUMask(ckt, instPtr, HICUM_QUEST_RCX_T, &rcx_t, select);
            HICUMask(ckt, instPtr, HICUM_QUEST_RE_T, &re_t, select);
            HICUMask(ckt, instPtr, HICUM_QUEST_RB, &rb, select);
            HICUMask(ckt, instPtr, HICUM_QUEST_BETAAC, &BETAAC, select);

            value->rValue = GMi.rValue/(
                2 * M_PI * (
                    CPIi.rValue + CPIx.rValue +
                    CMUi.rValue + CMUx.rValue +
                    (rcx_t.rValue + re_t.rValue + (re_t.rValue + rb.rValue)/BETAAC.rValue)
                ) * GMi.rValue * (
                    CMUi.rValue + CMUx.rValue
                )
            );
            return(OK);

/* power */
        case HICUM_QUEST_POWER:
            value->rValue = here->HICUMpterm;
            return(OK);
/* temperature */
        case HICUM_QUEST_TK:
            value->rValue = here->HICUMtemp;
            return(OK);
        case HICUM_QUEST_DTSH:
            value->rValue = here->HICUMdtemp_sh;
            return(OK);
        default:
            return(E_BADPARM);
    }
    /* NOTREACHED */
}

