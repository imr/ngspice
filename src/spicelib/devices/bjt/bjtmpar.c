/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

/*
 * This routine sets model parameters for
 * BJTs in the circuit.
 */

#include "ngspice/ngspice.h"
#include "ngspice/const.h"
#include "ngspice/ifsim.h"
#include "bjtdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
BJTmParam(int param, IFvalue *value, GENmodel *inModel)
{
    BJTmodel *mods = (BJTmodel*)inModel;

    switch(param) {
        case BJT_MOD_NPN:
            if(value->iValue) {
                mods->BJTtype = NPN;
            }
            break;
        case BJT_MOD_PNP:
            if(value->iValue) {
                mods->BJTtype = PNP;
            }
            break;
        case BJT_MOD_SUBS:
            mods->BJTsubs = value->iValue;
            mods->BJTsubsGiven = TRUE;
            break;
        case BJT_MOD_TNOM:
            mods->BJTtnom = value->rValue+CONSTCtoK;
            mods->BJTtnomGiven = TRUE;
            break;
        case BJT_MOD_IS:
            mods->BJTsatCur = value->rValue;
            mods->BJTsatCurGiven = TRUE;
            break;
        case BJT_MOD_IBE:
            mods->BJTBEsatCur = value->rValue;
            mods->BJTBEsatCurGiven = TRUE;
            break;
        case BJT_MOD_IBC:
            mods->BJTBCsatCur = value->rValue;
            mods->BJTBCsatCurGiven = TRUE;
            break;
        case BJT_MOD_BF:
            mods->BJTbetaF = value->rValue;
            mods->BJTbetaFGiven = TRUE;
            break;
        case BJT_MOD_NF:
            mods->BJTemissionCoeffF = value->rValue;
            mods->BJTemissionCoeffFGiven = TRUE;
            break;
        case BJT_MOD_VAF:
            mods->BJTearlyVoltF = value->rValue;
            mods->BJTearlyVoltFGiven = TRUE;
            break;
        case BJT_MOD_IKF:
            mods->BJTrollOffF = value->rValue;
            mods->BJTrollOffFGiven = TRUE;
            break;
        case BJT_MOD_ISE:
            mods->BJTleakBEcurrent = value->rValue;
            mods->BJTleakBEcurrentGiven = TRUE;
            break;
        case BJT_MOD_NE:
            mods->BJTleakBEemissionCoeff = value->rValue;
            mods->BJTleakBEemissionCoeffGiven = TRUE;
            break;
        case BJT_MOD_BR:
            mods->BJTbetaR = value->rValue;
            mods->BJTbetaRGiven = TRUE;
            break;
        case BJT_MOD_NR:
            mods->BJTemissionCoeffR = value->rValue;
            mods->BJTemissionCoeffRGiven = TRUE;
            break;
        case BJT_MOD_VAR:
            mods->BJTearlyVoltR = value->rValue;
            mods->BJTearlyVoltRGiven = TRUE;
            break;
        case BJT_MOD_IKR:
            mods->BJTrollOffR = value->rValue;
            mods->BJTrollOffRGiven = TRUE;
            break;
        case BJT_MOD_ISC:
            mods->BJTleakBCcurrent = value->rValue;
            mods->BJTleakBCcurrentGiven = TRUE;
            break;
        case BJT_MOD_NC:
            mods->BJTleakBCemissionCoeff = value->rValue;
            mods->BJTleakBCemissionCoeffGiven = TRUE;
            break;
        case BJT_MOD_RB:
            mods->BJTbaseResist = value->rValue;
            mods->BJTbaseResistGiven = TRUE;
            break;
        case BJT_MOD_IRB:
            mods->BJTbaseCurrentHalfResist = value->rValue;
            mods->BJTbaseCurrentHalfResistGiven = TRUE;
            break;
        case BJT_MOD_RBM:
            mods->BJTminBaseResist = value->rValue;
            mods->BJTminBaseResistGiven = TRUE;
            break;
        case BJT_MOD_RE:
            mods->BJTemitterResist = value->rValue;
            mods->BJTemitterResistGiven = TRUE;
            break;
        case BJT_MOD_RC:
            mods->BJTcollectorResist = value->rValue;
            mods->BJTcollectorResistGiven = TRUE;
            break;
        case BJT_MOD_CJE:
            mods->BJTdepletionCapBE = value->rValue;
            mods->BJTdepletionCapBEGiven = TRUE;
            break;
        case BJT_MOD_VJE:
            mods->BJTpotentialBE = value->rValue;
            mods->BJTpotentialBEGiven = TRUE;
            break;
        case BJT_MOD_MJE:
            mods->BJTjunctionExpBE = value->rValue;
            mods->BJTjunctionExpBEGiven = TRUE;
            break;
        case BJT_MOD_TF:
            mods->BJTtransitTimeF = value->rValue;
            mods->BJTtransitTimeFGiven = TRUE;
            break;
        case BJT_MOD_XTF:
            mods->BJTtransitTimeBiasCoeffF = value->rValue;
            mods->BJTtransitTimeBiasCoeffFGiven = TRUE;
            break;
        case BJT_MOD_VTF:
            mods->BJTtransitTimeFVBC = value->rValue;
            mods->BJTtransitTimeFVBCGiven = TRUE;
            break;
        case BJT_MOD_ITF:
            mods->BJTtransitTimeHighCurrentF = value->rValue;
            mods->BJTtransitTimeHighCurrentFGiven = TRUE;
            break;
        case BJT_MOD_PTF:
            mods->BJTexcessPhase = value->rValue;
            mods->BJTexcessPhaseGiven = TRUE;
            break;
        case BJT_MOD_CJC:
            mods->BJTdepletionCapBC = value->rValue;
            mods->BJTdepletionCapBCGiven = TRUE;
            break;
        case BJT_MOD_VJC:
            mods->BJTpotentialBC = value->rValue;
            mods->BJTpotentialBCGiven = TRUE;
            break;
        case BJT_MOD_MJC:
            mods->BJTjunctionExpBC = value->rValue;
            mods->BJTjunctionExpBCGiven = TRUE;
            break;
        case BJT_MOD_XCJC:
            mods->BJTbaseFractionBCcap = value->rValue;
            mods->BJTbaseFractionBCcapGiven = TRUE;
            break;
        case BJT_MOD_TR:
            mods->BJTtransitTimeR = value->rValue;
            mods->BJTtransitTimeRGiven = TRUE;
            break;
        case BJT_MOD_CJS:
            mods->BJTcapSub = value->rValue;
            mods->BJTcapSubGiven = TRUE;
            break;
        case BJT_MOD_VJS:
            mods->BJTpotentialSubstrate = value->rValue;
            mods->BJTpotentialSubstrateGiven = TRUE;
            break;
        case BJT_MOD_MJS:
            mods->BJTexponentialSubstrate = value->rValue;
            mods->BJTexponentialSubstrateGiven = TRUE;
            break;
        case BJT_MOD_XTB:
            mods->BJTbetaExp = value->rValue;
            mods->BJTbetaExpGiven = TRUE;
            break;
        case BJT_MOD_EG:
            mods->BJTenergyGap = value->rValue;
            mods->BJTenergyGapGiven = TRUE;
            break;
        case BJT_MOD_XTI:
            mods->BJTtempExpIS = value->rValue;
            mods->BJTtempExpISGiven = TRUE;
            break;
        case BJT_MOD_FC:
            mods->BJTdepletionCapCoeff = value->rValue;
            mods->BJTdepletionCapCoeffGiven = TRUE;
            break;
        case BJT_MOD_KF:
            mods->BJTfNcoef = value->rValue;
            mods->BJTfNcoefGiven = TRUE;
            break;
        case BJT_MOD_AF:
            mods->BJTfNexp = value->rValue;
            mods->BJTfNexpGiven = TRUE;
            break;
        case BJT_MOD_ISS:
            mods->BJTsubSatCur = value->rValue;
            mods->BJTsubSatCurGiven = TRUE;
            break;
        case BJT_MOD_NS:
            mods->BJTemissionCoeffS = value->rValue;
            mods->BJTemissionCoeffSGiven = TRUE;
            break;
        case BJT_MOD_RCO:
            mods->BJTintCollResist = value->rValue;
            mods->BJTintCollResistGiven = TRUE;
            break;
        case BJT_MOD_VO:
            mods->BJTepiSatVoltage = value->rValue;
            mods->BJTepiSatVoltageGiven = TRUE;
            break;
        case BJT_MOD_GAMMA:
            mods->BJTepiDoping = value->rValue;
            mods->BJTepiDopingGiven = TRUE;
            break;
        case BJT_MOD_QCO:
            mods->BJTepiCharge = value->rValue;
            mods->BJTepiChargeGiven = TRUE;
            break;
        case BJT_MOD_TLEV:
            mods->BJTtlev = value->iValue;
            mods->BJTtlevGiven = TRUE;
            break;
        case BJT_MOD_TLEVC:
            mods->BJTtlevc = value->iValue;
            mods->BJTtlevcGiven = TRUE;
            break;
        case BJT_MOD_TBF1:
            mods->BJTtbf1 = value->rValue;
            mods->BJTtbf1Given = TRUE;
            break;
        case BJT_MOD_TBF2:
            mods->BJTtbf2 = value->rValue;
            mods->BJTtbf2Given = TRUE;
            break;
        case BJT_MOD_TBR1:
            mods->BJTtbr1 = value->rValue;
            mods->BJTtbr1Given = TRUE;
            break;
        case BJT_MOD_TBR2:
            mods->BJTtbr2 = value->rValue;
            mods->BJTtbr2Given = TRUE;
            break;
        case BJT_MOD_TIKF1:
            mods->BJTtikf1 = value->rValue;
            mods->BJTtikf1Given = TRUE;
            break;
        case BJT_MOD_TIKF2:
            mods->BJTtikf2 = value->rValue;
            mods->BJTtikf2Given = TRUE;
            break;
        case BJT_MOD_TIKR1:
            mods->BJTtikr1 = value->rValue;
            mods->BJTtikr1Given = TRUE;
            break;
        case BJT_MOD_TIKR2:
            mods->BJTtikr2 = value->rValue;
            mods->BJTtikr2Given = TRUE;
            break;
        case BJT_MOD_TIRB1:
            mods->BJTtirb1 = value->rValue;
            mods->BJTtirb1Given = TRUE;
            break;
        case BJT_MOD_TIRB2:
            mods->BJTtirb2 = value->rValue;
            mods->BJTtirb2Given = TRUE;
            break;
        case BJT_MOD_TNC1:
            mods->BJTtnc1 = value->rValue;
            mods->BJTtnc1Given = TRUE;
            break;
        case BJT_MOD_TNC2:
            mods->BJTtnc2 = value->rValue;
            mods->BJTtnc2Given = TRUE;
            break;
        case BJT_MOD_TNE1:
            mods->BJTtne1 = value->rValue;
            mods->BJTtne1Given = TRUE;
            break;
        case BJT_MOD_TNE2:
            mods->BJTtne2 = value->rValue;
            mods->BJTtne2Given = TRUE;
            break;
        case BJT_MOD_TNF1:
            mods->BJTtnf1 = value->rValue;
            mods->BJTtnf1Given = TRUE;
            break;
        case BJT_MOD_TNF2:
            mods->BJTtnf2 = value->rValue;
            mods->BJTtnf2Given = TRUE;
            break;
        case BJT_MOD_TNR1:
            mods->BJTtnr1 = value->rValue;
            mods->BJTtnr1Given = TRUE;
            break;
        case BJT_MOD_TNR2:
            mods->BJTtnr2 = value->rValue;
            mods->BJTtnr2Given = TRUE;
            break;
        case BJT_MOD_TRB1:
            mods->BJTtrb1 = value->rValue;
            mods->BJTtrb1Given = TRUE;
            break;
        case BJT_MOD_TRB2:
            mods->BJTtrb2 = value->rValue;
            mods->BJTtrb2Given = TRUE;
            break;
        case BJT_MOD_TRC1:
            mods->BJTtrc1 = value->rValue;
            mods->BJTtrc1Given = TRUE;
            break;
        case BJT_MOD_TRC2:
            mods->BJTtrc2 = value->rValue;
            mods->BJTtrc2Given = TRUE;
            break;
        case BJT_MOD_TRE1:
            mods->BJTtre1 = value->rValue;
            mods->BJTtre1Given = TRUE;
            break;
        case BJT_MOD_TRE2:
            mods->BJTtre2 = value->rValue;
            mods->BJTtre2Given = TRUE;
            break;
        case BJT_MOD_TRM1:
            mods->BJTtrm1 = value->rValue;
            mods->BJTtrm1Given = TRUE;
            break;
        case BJT_MOD_TRM2:
            mods->BJTtrm2 = value->rValue;
            mods->BJTtrm2Given = TRUE;
            break;
        case BJT_MOD_TVAF1:
            mods->BJTtvaf1 = value->rValue;
            mods->BJTtvaf1Given = TRUE;
            break;
        case BJT_MOD_TVAF2:
            mods->BJTtvaf2 = value->rValue;
            mods->BJTtvaf2Given = TRUE;
            break;
        case BJT_MOD_TVAR1:
            mods->BJTtvar1 = value->rValue;
            mods->BJTtvar1Given = TRUE;
            break;
        case BJT_MOD_TVAR2:
            mods->BJTtvar2 = value->rValue;
            mods->BJTtvar2Given = TRUE;
            break;
        case BJT_MOD_CTC:
            mods->BJTctc = value->rValue;
            mods->BJTctcGiven = TRUE;
            break;
        case BJT_MOD_CTE:
            mods->BJTcte = value->rValue;
            mods->BJTcteGiven = TRUE;
            break;
        case BJT_MOD_CTS:
            mods->BJTcts = value->rValue;
            mods->BJTctsGiven = TRUE;
            break;
        case BJT_MOD_TVJE:
            mods->BJTtvje = value->rValue;
            mods->BJTtvjeGiven = TRUE;
            break;
        case BJT_MOD_TVJC:
            mods->BJTtvjc = value->rValue;
            mods->BJTtvjcGiven = TRUE;
            break;
        case BJT_MOD_TVJS:
            mods->BJTtvjs = value->rValue;
            mods->BJTtvjsGiven = TRUE;
            break;
        case BJT_MOD_TITF1:
            mods->BJTtitf1 = value->rValue;
            mods->BJTtitf1Given = TRUE;
            break;
        case BJT_MOD_TITF2:
            mods->BJTtitf2 = value->rValue;
            mods->BJTtitf2Given = TRUE;
            break;
        case BJT_MOD_TTF1:
            mods->BJTttf1 = value->rValue;
            mods->BJTttf1Given = TRUE;
            break;
        case BJT_MOD_TTF2:
            mods->BJTttf2 = value->rValue;
            mods->BJTttf2Given = TRUE;
            break;
        case BJT_MOD_TTR1:
            mods->BJTttr1 = value->rValue;
            mods->BJTttr1Given = TRUE;
            break;
        case BJT_MOD_TTR2:
            mods->BJTttr2 = value->rValue;
            mods->BJTttr2Given = TRUE;
            break;
        case BJT_MOD_TMJE1:
            mods->BJTtmje1 = value->rValue;
            mods->BJTtmje1Given = TRUE;
            break;
        case BJT_MOD_TMJE2:
            mods->BJTtmje2 = value->rValue;
            mods->BJTtmje2Given = TRUE;
            break;
        case BJT_MOD_TMJC1:
            mods->BJTtmjc1 = value->rValue;
            mods->BJTtmjc1Given = TRUE;
            break;
        case BJT_MOD_TMJC2:
            mods->BJTtmjc2 = value->rValue;
            mods->BJTtmjc2Given = TRUE;
            break;
        case BJT_MOD_TMJS1:
            mods->BJTtmjs1 = value->rValue;
            mods->BJTtmjs1Given = TRUE;
            break;
        case BJT_MOD_TMJS2:
            mods->BJTtmjs2 = value->rValue;
            mods->BJTtmjs2Given = TRUE;
            break;
        case BJT_MOD_TNS1:
            mods->BJTtns1 = value->rValue;
            mods->BJTtns1Given = TRUE;
            break;
        case BJT_MOD_TNS2:
            mods->BJTtns2 = value->rValue;
            mods->BJTtns2Given = TRUE;
            break;
        case BJT_MOD_NKF:
            mods->BJTnkf = value->rValue;
            mods->BJTnkfGiven = TRUE;
            break;
        case BJT_MOD_TIS1:
            mods->BJTtis1 = value->rValue;
            mods->BJTtis1Given = TRUE;
            break;
        case BJT_MOD_TIS2:
            mods->BJTtis2 = value->rValue;
            mods->BJTtis2Given = TRUE;
            break;
        case BJT_MOD_TISE1:
            mods->BJTtise1 = value->rValue;
            mods->BJTtise1Given = TRUE;
            break;
        case BJT_MOD_TISE2:
            mods->BJTtise2 = value->rValue;
            mods->BJTtise2Given = TRUE;
            break;
        case BJT_MOD_TISC1:
            mods->BJTtisc1 = value->rValue;
            mods->BJTtisc1Given = TRUE;
            break;
        case BJT_MOD_TISC2:
            mods->BJTtisc2 = value->rValue;
            mods->BJTtisc2Given = TRUE;
            break;
        case BJT_MOD_TISS1:
            mods->BJTtiss1 = value->rValue;
            mods->BJTtiss1Given = TRUE;
            break;
        case BJT_MOD_TISS2:
            mods->BJTtiss2 = value->rValue;
            mods->BJTtiss2Given = TRUE;
            break;
        case BJT_MOD_QUASIMOD:
            mods->BJTquasimod = value->iValue;
            mods->BJTquasimodGiven = TRUE;
            break;
        case BJT_MOD_EGQS:
            mods->BJTenergyGapQS = value->rValue;
            mods->BJTenergyGapQSGiven = TRUE;
            break;
        case BJT_MOD_XRCI:
            mods->BJTtempExpRCI = value->rValue;
            mods->BJTtempExpRCIGiven = TRUE;
            break;
        case BJT_MOD_XD:
            mods->BJTtempExpVO = value->rValue;
            mods->BJTtempExpVOGiven = TRUE;
            break;
        case BJT_MOD_VBE_MAX:
            mods->BJTvbeMax = value->rValue;
            mods->BJTvbeMaxGiven = TRUE;
            break;
        case BJT_MOD_VBC_MAX:
            mods->BJTvbcMax = value->rValue;
            mods->BJTvbcMaxGiven = TRUE;
            break;
        case BJT_MOD_VCE_MAX:
            mods->BJTvceMax = value->rValue;
            mods->BJTvceMaxGiven = TRUE;
            break;
        case BJT_MOD_IC_MAX:
            mods->BJTicMax = value->rValue;
            mods->BJTicMaxGiven = TRUE;
            break;
        case BJT_MOD_IB_MAX:
            mods->BJTibMax = value->rValue;
            mods->BJTibMaxGiven = TRUE;
            break;
        case BJT_MOD_PD_MAX:
            mods->BJTpdMax = value->rValue;
            mods->BJTpdMaxGiven = TRUE;
            break;
        case BJT_MOD_TE_MAX:
            mods->BJTteMax = value->rValue;
            mods->BJTteMaxGiven = TRUE;
            break;
        case BJT_MOD_RTH0:
            mods->BJTrth0 = value->rValue;
            mods->BJTrth0Given = TRUE;
            break;

        default:
            return(E_BADPARM);
    }
    return(OK);
}
