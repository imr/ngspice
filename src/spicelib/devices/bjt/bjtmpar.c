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

#include "ngspice.h"
#include "const.h"
#include "ifsim.h"
#include "bjtdefs.h"
#include "sperror.h"
#include "suffix.h"


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
        case BJT_MOD_TNOM:
            mods->BJTtnom = value->rValue+CONSTCtoK;
            mods->BJTtnomGiven = TRUE;
            break;
        case BJT_MOD_IS:
            mods->BJTsatCur = value->rValue;
            mods->BJTsatCurGiven = TRUE;
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
        case BJT_MOD_C2:
            mods->BJTc2 = value->rValue;
            mods->BJTc2Given=TRUE;
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
        case BJT_MOD_C4:
            mods->BJTc4 = value->rValue;
            mods->BJTc4Given=TRUE;
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
            mods->BJTcapCS = value->rValue;
            mods->BJTcapCSGiven = TRUE;
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
        default:
            return(E_BADPARM);
    }
    return(OK);
}
