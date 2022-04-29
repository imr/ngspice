/**********
License              : 3-clause BSD
Spice3 Implementation: 2019-2020 Dietmar Warning, Markus Müller, Mario Krattenmacher
Model Author         : 1990 Michael Schröter TU Dresden
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
        case HICUM_QUEST_VBBP:
            value->rValue = *(ckt->CKTstate0 + here->HICUMbaseNode)-*(ckt->CKTstate0 + here->HICUMbaseBPNode);
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
        case HICUM_QUEST_VBIEI:
            value->rValue = *(ckt->CKTstate0 + here->HICUMvbiei);
            return(OK);
        case HICUM_QUEST_VBPBI:
            value->rValue = *(ckt->CKTstate0 + here->HICUMvbpbi);
            return(OK);
        case HICUM_QUEST_VBICI:
            value->rValue = *(ckt->CKTstate0 + here->HICUMvbici);
            return(OK);
        case HICUM_QUEST_VCIEI:
            value->rValue = *(ckt->CKTstate0 + here->HICUMvbiei) - *(ckt->CKTstate0 + here->HICUMvbici);
            return(OK);
/* currents */
        case HICUM_QUEST_CC:
            value->rValue = *(ckt->CKTstate0 + here->HICUMiciei) -
                            *(ckt->CKTstate0 + here->HICUMibici) -
                            *(ckt->CKTstate0 + here->HICUMibpci) -
                            *(ckt->CKTstate0 + here->HICUMisici);
            value->rValue *= HICUMmodPtr(here)->HICUMtype;
            return(OK);
        case HICUM_QUEST_CB:
            value->rValue = *(ckt->CKTstate0 + here->HICUMibiei) +
                            *(ckt->CKTstate0 + here->HICUMibici) +
                            *(ckt->CKTstate0 + here->HICUMibpci) +
                            *(ckt->CKTstate0 + here->HICUMibpsi);
            value->rValue *= HICUMmodPtr(here)->HICUMtype;
            return(OK);
        case HICUM_QUEST_CE:
            value->rValue = - *(ckt->CKTstate0 + here->HICUMibiei) -
                             *(ckt->CKTstate0 + here->HICUMibpei) -
                             *(ckt->CKTstate0 + here->HICUMiciei);
            value->rValue *= HICUMmodPtr(here)->HICUMtype;
            return(OK);
        case HICUM_QUEST_CS:
            value->rValue = *(ckt->CKTstate0 + here->HICUMisici) -
                            *(ckt->CKTstate0 + here->HICUMibpsi);
            value->rValue *= HICUMmodPtr(here)->HICUMtype;
            return(OK);
        case HICUM_QUEST_CAVL:
            value->rValue = here->HICUMiavl;
            return(OK);
        case HICUM_QUEST_CBEI:
            value->rValue = *(ckt->CKTstate0 + here->HICUMibiei);
            return(OK);
        case HICUM_QUEST_CBCI:
            value->rValue = *(ckt->CKTstate0 + here->HICUMibici);
            return(OK);
/* resistances */
        case HICUM_QUEST_RCX_T:
            value->rValue = here->HICUMrcx_t.rpart;
            return(OK);
        case HICUM_QUEST_RE_T:
            value->rValue = here->HICUMre_t.rpart;
            return(OK);
        case HICUM_QUEST_IT:
            value->rValue = *(ckt->CKTstate0 + here->HICUMiciei);
            return(OK);
        case HICUM_QUEST_RBI:
            value->rValue = here->HICUMrbi;
            return(OK);
        case HICUM_QUEST_RB:
            value->rValue = here->HICUMrbi + here->HICUMrbx_t.rpart;
            return(OK);
/* transconductances and capacitances */
        case HICUM_QUEST_BETADC:
            HICUMask(ckt, instPtr, HICUM_QUEST_CC, &IC, select);
            HICUMask(ckt, instPtr, HICUM_QUEST_CB, &IB, select);
            if (IB.rValue != 0.0) {
                value->rValue = IC.rValue/IB.rValue;
            } else {
                value->rValue = 0.0;
            }
            return(OK);
        case HICUM_QUEST_GMI:
            value->rValue = *(ckt->CKTstate0 + here->HICUMiciei_Vbiei);
            return(OK);
        case HICUM_QUEST_GMS:
            value->rValue = *(ckt->CKTstate0 + here->HICUMibpsi_Vbpci);
            return(OK);
        case HICUM_QUEST_RPII:
            value->rValue = 1/( *(ckt->CKTstate0 + here->HICUMibiei_Vbiei) );
            return(OK);
        case HICUM_QUEST_RPIX:
            value->rValue = 1/( *(ckt->CKTstate0 + here->HICUMibpei_Vbpei) );
            return(OK);
        case HICUM_QUEST_RMUI:
            value->rValue = 1/( *(ckt->CKTstate0 + here->HICUMibici_Vbici) + ckt->CKTgmin);
            return(OK);
        case HICUM_QUEST_RMUX:
            value->rValue = 1/( *(ckt->CKTstate0 + here->HICUMibpci_Vbpci) + ckt->CKTgmin);
            return(OK);
        case HICUM_QUEST_ROI:
            value->rValue = 1/( *(ckt->CKTstate0 + here->HICUMiciei_Vbiei) + ckt->CKTgmin);
            return(OK);
        case HICUM_QUEST_CPII:
            value->rValue = here->HICUMcapjei + here->HICUMcapdeix;
            return(OK);
        case HICUM_QUEST_CPIX:
            value->rValue = here->HICUMcapjep + here->HICUMcbepar_scaled;
            return(OK);
        case HICUM_QUEST_CMUI:
            value->rValue = here->HICUMcapjci + here->HICUMcapdci;
            return(OK);
        case HICUM_QUEST_CMUX:
            value->rValue = here->HICUMcapjcx_t_i + here->HICUMcapjcx_t_ii + here->HICUMcbcpar_scaled + here->HICUMcapdsu;
            return(OK);
        case HICUM_QUEST_CCS:
            value->rValue = here->HICUMcapjs + here->HICUMcapscp;
            return(OK);
        case HICUM_QUEST_CRBI:
            value->rValue = here->HICUMcaprbi;
            return(OK);
        case HICUM_QUEST_BETAAC:
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
        case HICUM_QUEST_ICK:
            value->rValue = here->HICUMick;
            return(OK);
        case HICUM_QUEST_POWER:
            value->rValue = here->HICUMpterm;
            return(OK);
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
