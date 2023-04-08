/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1987 Mathew Lew and Thomas L. Quarles
Model Author: 1995 Colin McAndrew Motorola
Spice3 Implementation: 2003 Dietmar Warning DAnalyse GmbH
**********/

#include "ngspice/ngspice.h"
#include "ngspice/const.h"
#include "ngspice/ifsim.h"
#include "ngspice/cktdefs.h"
#include "ngspice/devdefs.h"
#include "vbicdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


/*ARGSUSED*/
int
VBICmAsk(CKTcircuit *ckt, GENmodel *instPtr, int which, IFvalue *value)
{
    VBICmodel *here = (VBICmodel*)instPtr;

    NG_IGNORE(ckt);

    switch(which) {
        case VBIC_MOD_TNOM:
            value->rValue = here->VBICtnom;
            return(OK);
        case VBIC_MOD_RCX:
            value->rValue = here->VBICextCollResist;
            return(OK);
        case VBIC_MOD_RCI:
            value->rValue = here->VBICintCollResist;
            return(OK);
        case VBIC_MOD_VO:
            value->rValue = here->VBICepiSatVoltage;
            return(OK);
        case VBIC_MOD_GAMM:
            value->rValue = here->VBICepiDoping;
            return(OK);
        case VBIC_MOD_HRCF:
            value->rValue = here->VBIChighCurFac;
            return(OK);
        case VBIC_MOD_RBX:
            value->rValue = here->VBICextBaseResist;
            return(OK);
        case VBIC_MOD_RBI:
            value->rValue = here->VBICintBaseResist;
            return(OK);
        case VBIC_MOD_RE:
            value->rValue = here->VBICemitterResist;
            return(OK);
        case VBIC_MOD_RS:
            value->rValue = here->VBICsubstrateResist;
            return(OK);
       case VBIC_MOD_RBP:
            value->rValue = here->VBICparBaseResist;
            return(OK);
        case VBIC_MOD_IS:
            value->rValue = here->VBICsatCur;
            return(OK);
        case VBIC_MOD_NF:
            value->rValue = here->VBICemissionCoeffF;
            return(OK);
        case VBIC_MOD_NR:
            value->rValue = here->VBICemissionCoeffR;
            return(OK);
        case VBIC_MOD_FC:
            value->rValue = here->VBICdeplCapLimitF;
            return(OK);
        case VBIC_MOD_CBEO:
            value->rValue = here->VBICextOverlapCapBE;
            return(OK);
        case VBIC_MOD_CJE:
            value->rValue = here->VBICdepletionCapBE;
            return(OK);
        case VBIC_MOD_PE:
            value->rValue = here->VBICpotentialBE;
            return(OK);
        case VBIC_MOD_ME:
            value->rValue = here->VBICjunctionExpBE;
            return(OK);
        case VBIC_MOD_AJE:
            value->rValue = here->VBICsmoothCapBE;
            return(OK);
       case VBIC_MOD_CBCO:
            value->rValue = here->VBICextOverlapCapBC;
            return(OK);
        case VBIC_MOD_CJC:
            value->rValue = here->VBICdepletionCapBC;
            return(OK);
        case VBIC_MOD_QCO:
            value->rValue = here->VBICepiCharge;
            return(OK);
       case VBIC_MOD_CJEP:
            value->rValue = here->VBICextCapBC;
            return(OK);
        case VBIC_MOD_PC:
            value->rValue = here->VBICpotentialBC;
            return(OK);
        case VBIC_MOD_MC:
            value->rValue = here->VBICjunctionExpBC;
            return(OK);
        case VBIC_MOD_AJC:
            value->rValue = here->VBICsmoothCapBC;
            return(OK);
        case VBIC_MOD_CJCP:
            value->rValue = here->VBICextCapSC;
            return(OK);
        case VBIC_MOD_PS:
            value->rValue = here->VBICpotentialSC;
            return(OK);
       case VBIC_MOD_MS:
            value->rValue = here->VBICjunctionExpSC;
            return(OK);
       case VBIC_MOD_AJS:
            value->rValue = here->VBICsmoothCapSC;
            return(OK);
        case VBIC_MOD_IBEI:
            value->rValue = here->VBICidealSatCurBE;
            return(OK);
        case VBIC_MOD_WBE:
            value->rValue = here->VBICportionIBEI;
            return(OK);
        case VBIC_MOD_NEI:
            value->rValue = here->VBICidealEmissCoeffBE;
            return(OK);
        case VBIC_MOD_IBEN:
            value->rValue = here->VBICnidealSatCurBE;
            return(OK);
        case VBIC_MOD_NEN:
            value->rValue = here->VBICnidealEmissCoeffBE;
            return(OK);
        case VBIC_MOD_IBCI:
            value->rValue = here->VBICidealSatCurBC;
            return(OK);
        case VBIC_MOD_NCI:
            value->rValue = here->VBICidealEmissCoeffBC;
            return(OK);
        case VBIC_MOD_IBCN:
            value->rValue = here->VBICnidealSatCurBC;
            return(OK);
        case VBIC_MOD_NCN:
            value->rValue = here->VBICnidealEmissCoeffBC;
            return(OK);
       case VBIC_MOD_AVC1:
            value->rValue = here->VBICavalanchePar1BC;
            return(OK);
        case VBIC_MOD_AVC2:
            value->rValue = here->VBICavalanchePar2BC;
            return(OK);
        case VBIC_MOD_ISP:
            value->rValue = here->VBICparasitSatCur;
            return(OK);
        case VBIC_MOD_WSP:
            value->rValue = here->VBICportionICCP;
            return(OK);
        case VBIC_MOD_NFP:
            value->rValue = here->VBICparasitFwdEmissCoeff;
            return(OK);
        case VBIC_MOD_IBEIP:
            value->rValue = here->VBICidealParasitSatCurBE;
            return(OK);
        case VBIC_MOD_IBENP:
            value->rValue = here->VBICnidealParasitSatCurBE;
            return(OK);
        case VBIC_MOD_IBCIP:
            value->rValue = here->VBICidealParasitSatCurBC;
            return(OK);
        case VBIC_MOD_NCIP:
            value->rValue = here->VBICidealParasitEmissCoeffBC;
            return(OK);
        case VBIC_MOD_IBCNP:
            value->rValue = here->VBICnidealParasitSatCurBC;
            return(OK);
        case VBIC_MOD_NCNP:
            value->rValue = here->VBICnidealParasitEmissCoeffBC;
            return(OK);
        case VBIC_MOD_VEF:
            value->rValue = here->VBICearlyVoltF;
            return(OK);
        case VBIC_MOD_VER:
            value->rValue = here->VBICearlyVoltR;
            return(OK);
        case VBIC_MOD_IKF:
            value->rValue = here->VBICrollOffF;
            return(OK);
        case VBIC_MOD_IKR:
            value->rValue = here->VBICrollOffR;
            return(OK);
        case VBIC_MOD_IKP:
            value->rValue = here->VBICparRollOff;
            return(OK);
        case VBIC_MOD_TF:
            value->rValue = here->VBICtransitTimeF;
            return(OK);
        case VBIC_MOD_QTF:
            value->rValue = here->VBICvarTransitTimeF;
            return(OK);
        case VBIC_MOD_XTF:
            value->rValue = here->VBICtransitTimeBiasCoeffF;
            return(OK);
        case VBIC_MOD_VTF:
            value->rValue = here->VBICtransitTimeFVBC;
            return(OK);
        case VBIC_MOD_ITF:
            value->rValue = here->VBICtransitTimeHighCurrentF;
            return(OK);
        case VBIC_MOD_TR:
            value->rValue = here->VBICtransitTimeR;
            return(OK);
        case VBIC_MOD_TD:
            value->rValue = here->VBICdelayTimeF;
            return(OK);
        case VBIC_MOD_KFN:
            value->rValue = here->VBICfNcoef;
            return(OK);
        case VBIC_MOD_AFN:
            value->rValue = here->VBICfNexpA;
            return(OK);
        case VBIC_MOD_BFN:
            value->rValue = here->VBICfNexpB;
            return(OK);
        case VBIC_MOD_XRE:
            value->rValue = here->VBICtempExpRE;
            return(OK);
        case VBIC_MOD_XRBI:
            value->rValue = here->VBICtempExpRBI;
            return(OK);
        case VBIC_MOD_XRCI:
            value->rValue = here->VBICtempExpRCI;
            return(OK);
        case VBIC_MOD_XRS:
            value->rValue = here->VBICtempExpRS;
            return(OK);
        case VBIC_MOD_XVO:
            value->rValue = here->VBICtempExpVO;
            return(OK);
        case VBIC_MOD_EA:
            value->rValue = here->VBICactivEnergyEA;
            return(OK);
        case VBIC_MOD_EAIE:
            value->rValue = here->VBICactivEnergyEAIE;
            return(OK);
        case VBIC_MOD_EAIC:
            value->rValue = here->VBICactivEnergyEAIC;
            return(OK);
        case VBIC_MOD_EAIS:
            value->rValue = here->VBICactivEnergyEAIS;
            return(OK);
        case VBIC_MOD_EANE:
            value->rValue = here->VBICactivEnergyEANE;
            return(OK);
        case VBIC_MOD_EANC:
            value->rValue = here->VBICactivEnergyEANC;
            return(OK);
        case VBIC_MOD_EANS:
            value->rValue = here->VBICactivEnergyEANS;
            return(OK);
        case VBIC_MOD_XIS:
            value->rValue = here->VBICtempExpIS;
            return(OK);
        case VBIC_MOD_XII:
            value->rValue = here->VBICtempExpII;
            return(OK);
        case VBIC_MOD_XIN:
            value->rValue = here->VBICtempExpIN;
            return(OK);
        case VBIC_MOD_TNF:
            value->rValue = here->VBICtempExpNF;
            return(OK);
        case VBIC_MOD_TAVC:
            value->rValue = here->VBICtempExpAVC;
            return(OK);
        case VBIC_MOD_RTH:
            value->rValue = here->VBICthermalResist;
            return(OK);
        case VBIC_MOD_CTH:
            value->rValue = here->VBICthermalCapacitance;
            return(OK);
        case VBIC_MOD_VRT:
            value->rValue = here->VBICpunchThroughVoltageBC;
            return(OK);
        case VBIC_MOD_ART:
            value->rValue = here->VBICdeplCapCoeff1;
            return(OK);
        case VBIC_MOD_CCSO:
            value->rValue = here->VBICfixedCapacitanceCS;
            return(OK);
        case VBIC_MOD_QBM:
            value->rValue = here->VBICsgpQBselector;
            return(OK);
        case VBIC_MOD_NKF:
            value->rValue = here->VBIChighCurrentBetaRolloff;
            return(OK);
        case VBIC_MOD_XIKF:
            value->rValue = here->VBICtempExpIKF;
            return(OK);
        case VBIC_MOD_XRCX:
            value->rValue = here->VBICtempExpRCX;
            return(OK);
        case VBIC_MOD_XRBX:
            value->rValue = here->VBICtempExpRBX;
            return(OK);
        case VBIC_MOD_XRBP:
            value->rValue = here->VBICtempExpRBP;
            return(OK);
        case VBIC_MOD_ISRR:
            value->rValue = here->VBICsepISRR;
            return(OK);
        case VBIC_MOD_XISR:
            value->rValue = here->VBICtempExpXISR;
            return(OK);
        case VBIC_MOD_DEAR:
            value->rValue = here->VBICdear;
            return(OK);
        case VBIC_MOD_EAP:
            value->rValue = here->VBICeap;
            return(OK);
        case VBIC_MOD_VBBE:
            value->rValue = here->VBICvbbe;
            return(OK);
        case VBIC_MOD_NBBE:
            value->rValue = here->VBICnbbe;
            return(OK);
        case VBIC_MOD_IBBE:
            value->rValue = here->VBICibbe;
            return(OK);
        case VBIC_MOD_TVBBE1:
            value->rValue = here->VBICtvbbe1;
            return(OK);
        case VBIC_MOD_TVBBE2:
            value->rValue = here->VBICtvbbe2;
            return(OK);
        case VBIC_MOD_TNBBE:
            value->rValue = here->VBICtnbbe;
            return(OK);
        case VBIC_MOD_EBBE:
            value->rValue = here->VBICebbe;
            return(OK);
        case VBIC_MOD_DTEMP:
            value->rValue = here->VBIClocTempDiff;
            return(OK);
        case VBIC_MOD_VERS:
            value->rValue = here->VBICrevVersion;
            return(OK);
        case VBIC_MOD_VREF:
            value->rValue = here->VBICrefVersion;
            return(OK);
        case VBIC_MOD_VBE_MAX:
            value->rValue = here->VBICvbeMax;
            return(OK);
        case VBIC_MOD_VBC_MAX:
            value->rValue = here->VBICvbcMax;
            return(OK);
        case VBIC_MOD_VCE_MAX:
            value->rValue = here->VBICvceMax;
            return(OK);
        case VBIC_MOD_VSUB_MAX:
            value->rValue = here->VBICvsubMax;
            return(OK);
        case VBIC_MOD_VBEFWD_MAX:
            value->rValue = here->VBICvbefwdMax;
            return(OK);
        case VBIC_MOD_VBCFWD_MAX:
            value->rValue = here->VBICvbcfwdMax;
            return(OK);
        case VBIC_MOD_VSUBFWD_MAX:
            value->rValue = here->VBICvsubfwdMax;
            return(OK);
        case VBIC_MOD_TYPE:
            if (here->VBICtype == NPN)
                value->sValue = "npn";
            else
                value->sValue = "pnp";
            return(OK);
        case VBIC_MOD_SELFT:
            value->iValue = here->VBICselft;
            return(OK);
        default:
            return(E_BADPARM);
    }
    /* NOTREACHED */
}

