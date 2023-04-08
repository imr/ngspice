/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Model Author: 1995 Colin McAndrew Motorola
Spice3 Implementation: 2003 Dietmar Warning DAnalyse GmbH
**********/

/*
 * This routine sets model parameters for
 * VBICs in the circuit.
 */

#include "ngspice/ngspice.h"
#include "ngspice/const.h"
#include "ngspice/ifsim.h"
#include "vbicdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

#define R_MIN 0.01

int
VBICmParam(int param, IFvalue *value, GENmodel *inModel)
{
    VBICmodel *mods = (VBICmodel*)inModel;

    switch(param) {
        case VBIC_MOD_NPN:
            if(value->iValue) {
                mods->VBICtype = NPN;
            }
            break;
        case VBIC_MOD_PNP:
            if(value->iValue) {
                mods->VBICtype = PNP;
            }
            break;

        case VBIC_MOD_TNOM:
            mods->VBICtnom = value->rValue;
            mods->VBICtnomGiven = TRUE;
            break;
        case VBIC_MOD_RCX:
            mods->VBICextCollResist = value->rValue;
            if (mods->VBICextCollResist < R_MIN) {
                mods->VBICextCollResist = R_MIN;
            }
            mods->VBICextCollResistGiven = TRUE;
            break;
        case VBIC_MOD_RCI:
            mods->VBICintCollResist = value->rValue;
            if (mods->VBICintCollResist < R_MIN) {
                mods->VBICintCollResist = R_MIN;
            }
            mods->VBICintCollResistGiven = TRUE;
            break;
        case VBIC_MOD_VO:
            mods->VBICepiSatVoltage = value->rValue;
            mods->VBICepiSatVoltageGiven = TRUE;
            break;
        case VBIC_MOD_GAMM:
            mods->VBICepiDoping = value->rValue;
            mods->VBICepiDopingGiven = TRUE;
            break;
        case VBIC_MOD_HRCF:
            mods->VBIChighCurFac = value->rValue;
            mods->VBIChighCurFacGiven = TRUE;
            break;
        case VBIC_MOD_RBX:
            mods->VBICextBaseResist = value->rValue;
            if (mods->VBICextBaseResist < R_MIN) {
                mods->VBICextBaseResist = R_MIN;
            }
            mods->VBICextBaseResistGiven = TRUE;
            break;
        case VBIC_MOD_RBI:
            mods->VBICintBaseResist = value->rValue;
            if (mods->VBICintBaseResist < R_MIN) {
                mods->VBICintBaseResist = R_MIN;
            }
            mods->VBICintBaseResistGiven = TRUE;
            break;
        case VBIC_MOD_RE:
            mods->VBICemitterResist = value->rValue;
            if (mods->VBICemitterResist < R_MIN) {
                mods->VBICemitterResist = R_MIN;
            }
            mods->VBICemitterResistGiven = TRUE;
            break;
        case VBIC_MOD_RS:
            mods->VBICsubstrateResist = value->rValue;
            if (mods->VBICsubstrateResist < R_MIN) {
                mods->VBICsubstrateResist = R_MIN;
            }
            mods->VBICsubstrateResistGiven = TRUE;
            break;
        case VBIC_MOD_RBP:
            mods->VBICparBaseResist = value->rValue;
            if (mods->VBICparBaseResist < R_MIN) {
                mods->VBICparBaseResist = R_MIN;
            }
            mods->VBICparBaseResistGiven = TRUE;
            break;
        case VBIC_MOD_IS:
            mods->VBICsatCur = value->rValue;
            mods->VBICsatCurGiven = TRUE;
            break;
        case VBIC_MOD_NF:
            mods->VBICemissionCoeffF = value->rValue;
            mods->VBICemissionCoeffFGiven = TRUE;
            break;
        case VBIC_MOD_NR:
            mods->VBICemissionCoeffR = value->rValue;
            mods->VBICemissionCoeffRGiven = TRUE;
            break;
        case VBIC_MOD_FC:
            mods->VBICdeplCapLimitF = value->rValue;
            mods->VBICdeplCapLimitFGiven = TRUE;
            break;
        case VBIC_MOD_CBEO:
            mods->VBICextOverlapCapBE = value->rValue;
            mods->VBICextOverlapCapBEGiven=TRUE;
            break;
        case VBIC_MOD_CJE:
            mods->VBICdepletionCapBE = value->rValue;
            mods->VBICdepletionCapBEGiven = TRUE;
            break;
        case VBIC_MOD_PE:
            mods->VBICpotentialBE = value->rValue;
            mods->VBICpotentialBEGiven = TRUE;
            break;
        case VBIC_MOD_ME:
            mods->VBICjunctionExpBE = value->rValue;
            mods->VBICjunctionExpBEGiven = TRUE;
            break;
        case VBIC_MOD_AJE:
            mods->VBICsmoothCapBE = value->rValue;
            mods->VBICsmoothCapBEGiven = TRUE;
            break;
        case VBIC_MOD_CBCO:
            mods->VBICextOverlapCapBC = value->rValue;
            mods->VBICextOverlapCapBCGiven=TRUE;
            break;
        case VBIC_MOD_CJC:
            mods->VBICdepletionCapBC = value->rValue;
            mods->VBICdepletionCapBCGiven = TRUE;
            break;
        case VBIC_MOD_QCO:
            mods->VBICepiCharge = value->rValue;
            mods->VBICepiChargeGiven = TRUE;
            break;
        case VBIC_MOD_CJEP:
            mods->VBICextCapBC = value->rValue;
            mods->VBICextCapBCGiven = TRUE;
            break;
        case VBIC_MOD_PC:
            mods->VBICpotentialBC = value->rValue;
            mods->VBICpotentialBCGiven = TRUE;
            break;
        case VBIC_MOD_MC:
            mods->VBICjunctionExpBC = value->rValue;
            mods->VBICjunctionExpBCGiven = TRUE;
            break;
        case VBIC_MOD_AJC:
            mods->VBICsmoothCapBC = value->rValue;
            mods->VBICsmoothCapBCGiven = TRUE;
            break;
        case VBIC_MOD_CJCP:
            mods->VBICextCapSC = value->rValue;
            mods->VBICextCapSCGiven = TRUE;
            break;
        case VBIC_MOD_PS:
            mods->VBICpotentialSC = value->rValue;
            mods->VBICpotentialSCGiven = TRUE;
            break;
        case VBIC_MOD_MS:
            mods->VBICjunctionExpSC = value->rValue;
            mods->VBICjunctionExpSCGiven = TRUE;
            break;
        case VBIC_MOD_AJS:
            mods->VBICsmoothCapSC = value->rValue;
            mods->VBICsmoothCapSCGiven = TRUE;
            break;
        case VBIC_MOD_IBEI:
            mods->VBICidealSatCurBE = value->rValue;
            mods->VBICidealSatCurBEGiven = TRUE;
            break;
        case VBIC_MOD_WBE:
            mods->VBICportionIBEI = value->rValue;
            mods->VBICportionIBEIGiven = TRUE;
            break;
        case VBIC_MOD_NEI:
            mods->VBICidealEmissCoeffBE = value->rValue;
            mods->VBICidealEmissCoeffBEGiven = TRUE;
            break;
        case VBIC_MOD_IBEN:
            mods->VBICnidealSatCurBE = value->rValue;
            mods->VBICnidealSatCurBEGiven = TRUE;
            break;
        case VBIC_MOD_NEN:
            mods->VBICnidealEmissCoeffBE = value->rValue;
            mods->VBICnidealEmissCoeffBEGiven = TRUE;
            break;
        case VBIC_MOD_IBCI:
            mods->VBICidealSatCurBC = value->rValue;
            mods->VBICidealSatCurBCGiven = TRUE;
            break;
        case VBIC_MOD_NCI:
            mods->VBICidealEmissCoeffBC = value->rValue;
            mods->VBICidealEmissCoeffBCGiven = TRUE;
            break;
        case VBIC_MOD_IBCN:
            mods->VBICnidealSatCurBC = value->rValue;
            mods->VBICnidealSatCurBCGiven = TRUE;
            break;
        case VBIC_MOD_NCN:
            mods->VBICnidealEmissCoeffBC = value->rValue;
            mods->VBICnidealEmissCoeffBCGiven = TRUE;
            break;
        case VBIC_MOD_AVC1:
            mods->VBICavalanchePar1BC = value->rValue;
            mods->VBICavalanchePar1BCGiven = TRUE;
            break;
        case VBIC_MOD_AVC2:
            mods->VBICavalanchePar2BC = value->rValue;
            mods->VBICavalanchePar2BCGiven = TRUE;
            break;
        case VBIC_MOD_ISP:
            mods->VBICparasitSatCur = value->rValue;
            mods->VBICparasitSatCurGiven = TRUE;
            break;
        case VBIC_MOD_WSP:
            mods->VBICportionICCP = value->rValue;
            mods->VBICportionICCPGiven = TRUE;
            break;
        case VBIC_MOD_NFP:
            mods->VBICparasitFwdEmissCoeff = value->rValue;
            mods->VBICparasitFwdEmissCoeffGiven = TRUE;
            break;
        case VBIC_MOD_IBEIP:
            mods->VBICidealParasitSatCurBE = value->rValue;
            mods->VBICidealParasitSatCurBEGiven = TRUE;
            break;
        case VBIC_MOD_IBENP:
            mods->VBICnidealParasitSatCurBE = value->rValue;
            mods->VBICnidealParasitSatCurBEGiven = TRUE;
            break;
        case VBIC_MOD_IBCIP:
            mods->VBICidealParasitSatCurBC = value->rValue;
            mods->VBICidealParasitSatCurBCGiven = TRUE;
            break;
        case VBIC_MOD_NCIP:
            mods->VBICidealParasitEmissCoeffBC = value->rValue;
            mods->VBICidealParasitEmissCoeffBCGiven = TRUE;
            break;
        case VBIC_MOD_IBCNP:
            mods->VBICnidealParasitSatCurBC = value->rValue;
            mods->VBICnidealParasitSatCurBCGiven = TRUE;
            break;
        case VBIC_MOD_NCNP:
            mods->VBICnidealParasitEmissCoeffBC = value->rValue;
            mods->VBICnidealParasitEmissCoeffBCGiven = TRUE;
            break;
        case VBIC_MOD_VEF:
            mods->VBICearlyVoltF = value->rValue;
            mods->VBICearlyVoltFGiven = TRUE;
            break;
        case VBIC_MOD_VER:
            mods->VBICearlyVoltR = value->rValue;
            mods->VBICearlyVoltRGiven = TRUE;
            break;
        case VBIC_MOD_IKF:
            mods->VBICrollOffF = value->rValue;
            mods->VBICrollOffFGiven = TRUE;
            break;
        case VBIC_MOD_IKR:
            mods->VBICrollOffR = value->rValue;
            mods->VBICrollOffRGiven = TRUE;
            break;
        case VBIC_MOD_IKP:
            mods->VBICparRollOff = value->rValue;
            mods->VBICparRollOffGiven = TRUE;
            break;
        case VBIC_MOD_TF:
            mods->VBICtransitTimeF = value->rValue;
            mods->VBICtransitTimeFGiven = TRUE;
            break;
        case VBIC_MOD_QTF:
            mods->VBICvarTransitTimeF = value->rValue;
            mods->VBICvarTransitTimeFGiven = TRUE;
            break;
        case VBIC_MOD_XTF:
            mods->VBICtransitTimeBiasCoeffF = value->rValue;
            mods->VBICtransitTimeBiasCoeffFGiven = TRUE;
            break;
        case VBIC_MOD_VTF:
            mods->VBICtransitTimeFVBC = value->rValue;
            mods->VBICtransitTimeFVBCGiven = TRUE;
            break;
        case VBIC_MOD_ITF:
            mods->VBICtransitTimeHighCurrentF = value->rValue;
            mods->VBICtransitTimeHighCurrentFGiven = TRUE;
            break;
        case VBIC_MOD_TR:
            mods->VBICtransitTimeR = value->rValue;
            mods->VBICtransitTimeRGiven = TRUE;
            break;
        case VBIC_MOD_TD:
            mods->VBICdelayTimeF = value->rValue;
            mods->VBICdelayTimeFGiven = TRUE;
            break;
        case VBIC_MOD_KFN:
            mods->VBICfNcoef = value->rValue;
            mods->VBICfNcoefGiven = TRUE;
            break;
        case VBIC_MOD_AFN:
            mods->VBICfNexpA = value->rValue;
            mods->VBICfNexpAGiven = TRUE;
            break;
        case VBIC_MOD_BFN:
            mods->VBICfNexpB = value->rValue;
            mods->VBICfNexpBGiven = TRUE;
            break;
        case VBIC_MOD_XRE:
            mods->VBICtempExpRE = value->rValue;
            mods->VBICtempExpREGiven = TRUE;
            break;
        case VBIC_MOD_XRBI:
            mods->VBICtempExpRBI = value->rValue;
            mods->VBICtempExpRBIGiven = TRUE;
            break;
        case VBIC_MOD_XRCI:
            mods->VBICtempExpRCI = value->rValue;
            mods->VBICtempExpRCIGiven = TRUE;
            break;
        case VBIC_MOD_XRS:
            mods->VBICtempExpRS = value->rValue;
            mods->VBICtempExpRSGiven = TRUE;
            break;
        case VBIC_MOD_XVO:
            mods->VBICtempExpVO = value->rValue;
            mods->VBICtempExpVOGiven = TRUE;
            break;
        case VBIC_MOD_EA:
            mods->VBICactivEnergyEA = value->rValue;
            mods->VBICactivEnergyEAGiven = TRUE;
            break;
        case VBIC_MOD_EAIE:
            mods->VBICactivEnergyEAIE = value->rValue;
            mods->VBICactivEnergyEAIEGiven = TRUE;
            break;
        case VBIC_MOD_EAIC:
            mods->VBICactivEnergyEAIC = value->rValue;
            mods->VBICactivEnergyEAICGiven = TRUE;
            break;
        case VBIC_MOD_EAIS:
            mods->VBICactivEnergyEAIS = value->rValue;
            mods->VBICactivEnergyEAISGiven = TRUE;
            break;
        case VBIC_MOD_EANE:
            mods->VBICactivEnergyEANE = value->rValue;
            mods->VBICactivEnergyEANEGiven = TRUE;
            break;
        case VBIC_MOD_EANC:
            mods->VBICactivEnergyEANC = value->rValue;
            mods->VBICactivEnergyEANCGiven = TRUE;
            break;
        case VBIC_MOD_EANS:
            mods->VBICactivEnergyEANS = value->rValue;
            mods->VBICactivEnergyEANSGiven = TRUE;
            break;
        case VBIC_MOD_XIS:
            mods->VBICtempExpIS = value->rValue;
            mods->VBICtempExpISGiven = TRUE;
            break;
        case VBIC_MOD_XII:
            mods->VBICtempExpII = value->rValue;
            mods->VBICtempExpIIGiven = TRUE;
            break;
        case VBIC_MOD_XIN:
            mods->VBICtempExpIN = value->rValue;
            mods->VBICtempExpINGiven = TRUE;
            break;
        case VBIC_MOD_TNF:
            mods->VBICtempExpNF = value->rValue;
            mods->VBICtempExpNFGiven = TRUE;
            break;
        case VBIC_MOD_TAVC:
            mods->VBICtempExpAVC = value->rValue;
            mods->VBICtempExpAVCGiven = TRUE;
            break;
        case VBIC_MOD_RTH:
            mods->VBICthermalResist = value->rValue;
            mods->VBICthermalResistGiven = TRUE;
            break;
        case VBIC_MOD_CTH:
            mods->VBICthermalCapacitance = value->rValue;
            mods->VBICthermalCapacitanceGiven = TRUE;
            break;
        case VBIC_MOD_VRT:
            mods->VBICpunchThroughVoltageBC = value->rValue;
            mods->VBICpunchThroughVoltageBCGiven = TRUE;
            break;
        case VBIC_MOD_ART:
            mods->VBICdeplCapCoeff1 = value->rValue;
            mods->VBICdeplCapCoeff1Given = TRUE;
            break;
        case VBIC_MOD_CCSO:
            mods->VBICfixedCapacitanceCS = value->rValue;
            mods->VBICfixedCapacitanceCSGiven = TRUE;
            break;
        case VBIC_MOD_QBM:
            mods->VBICsgpQBselector = value->rValue;
            mods->VBICsgpQBselectorGiven = TRUE;
            break;
        case VBIC_MOD_NKF:
            mods->VBIChighCurrentBetaRolloff = value->rValue;
            mods->VBIChighCurrentBetaRolloffGiven = TRUE;
            break;
        case VBIC_MOD_XIKF:
            mods->VBICtempExpIKF = value->rValue;
            mods->VBICtempExpIKFGiven = TRUE;
            break;
        case VBIC_MOD_XRCX:
            mods->VBICtempExpRCX = value->rValue;
            mods->VBICtempExpRCXGiven = TRUE;
            break;
        case VBIC_MOD_XRBX:
            mods->VBICtempExpRBX = value->rValue;
            mods->VBICtempExpRBXGiven = TRUE;
            break;
        case VBIC_MOD_XRBP:
            mods->VBICtempExpRBP = value->rValue;
            mods->VBICtempExpRBPGiven = TRUE;
            break;
        case VBIC_MOD_ISRR:
            mods->VBICsepISRR = value->rValue;
            mods->VBICsepISRRGiven = TRUE;
            break;
        case VBIC_MOD_XISR:
            mods->VBICtempExpXISR = value->rValue;
            mods->VBICtempExpXISRGiven = TRUE;
            break;
        case VBIC_MOD_DEAR:
            mods->VBICdear = value->rValue;
            mods->VBICdearGiven = TRUE;
            break;
        case VBIC_MOD_EAP:
            mods->VBICeap = value->rValue;
            mods->VBICeapGiven = TRUE;
            break;
        case VBIC_MOD_VBBE:
            mods->VBICvbbe = value->rValue;
            mods->VBICvbbeGiven = TRUE;
            break;
        case VBIC_MOD_NBBE:
            mods->VBICnbbe = value->rValue;
            mods->VBICnbbeGiven = TRUE;
            break;
        case VBIC_MOD_IBBE:
            mods->VBICibbe = value->rValue;
            mods->VBICibbeGiven = TRUE;
            break;
        case VBIC_MOD_TVBBE1:
            mods->VBICtvbbe1 = value->rValue;
            mods->VBICtvbbe1Given = TRUE;
            break;
        case VBIC_MOD_TVBBE2:
            mods->VBICtvbbe2 = value->rValue;
            mods->VBICtvbbe2Given = TRUE;
            break;
        case VBIC_MOD_TNBBE:
            mods->VBICtnbbe = value->rValue;
            mods->VBICtnbbeGiven = TRUE;
            break;
        case VBIC_MOD_EBBE:
            mods->VBICebbe = value->rValue;
            mods->VBICebbeGiven = TRUE;
            break;
        case VBIC_MOD_DTEMP:
            mods->VBIClocTempDiff = value->rValue;
            mods->VBIClocTempDiffGiven = TRUE;
            break;
        case VBIC_MOD_VERS:
            mods->VBICrevVersion = value->rValue;
            mods->VBICrevVersionGiven = TRUE;
            break;
        case VBIC_MOD_VREF:
            mods->VBICrefVersion = value->rValue;
            mods->VBICrefVersionGiven = TRUE;
            break;
        case VBIC_MOD_VBE_MAX:
            mods->VBICvbeMax = value->rValue;
            mods->VBICvbeMaxGiven = TRUE;
            break;
        case VBIC_MOD_VBC_MAX:
            mods->VBICvbcMax = value->rValue;
            mods->VBICvbcMaxGiven = TRUE;
            break;
        case VBIC_MOD_VCE_MAX:
            mods->VBICvceMax = value->rValue;
            mods->VBICvceMaxGiven = TRUE;
            break;
        case VBIC_MOD_VSUB_MAX:
            mods->VBICvsubMax = value->rValue;
            mods->VBICvsubMaxGiven = TRUE;
            break;
        case VBIC_MOD_VBEFWD_MAX:
            mods->VBICvbefwdMax = value->rValue;
            mods->VBICvbefwdMaxGiven = TRUE;
            break;
        case VBIC_MOD_VBCFWD_MAX:
            mods->VBICvbcfwdMax = value->rValue;
            mods->VBICvbcfwdMaxGiven = TRUE;
            break;
        case VBIC_MOD_VSUBFWD_MAX:
            mods->VBICvsubfwdMax = value->rValue;
            mods->VBICvsubfwdMaxGiven = TRUE;
            break;
        case VBIC_MOD_SELFT:
            mods->VBICselft = value->iValue;
            mods->VBICselftGiven = TRUE;
            break;
        default:
            return(E_BADPARM);
    }
    return(OK);
}
