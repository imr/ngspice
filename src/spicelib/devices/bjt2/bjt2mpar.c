/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: Alan Gillespie
**********/
/*
 */

/*
 * This routine sets model parameters for
 * BJT2s in the circuit.
 */

#include "ngspice.h"
#include "const.h"
#include "ifsim.h"
#include "bjt2defs.h"
#include "sperror.h"
#include "suffix.h"


int
BJT2mParam(int param, IFvalue *value, GENmodel *inModel)
{
    BJT2model *mods = (BJT2model*)inModel;

    switch(param) {
        case BJT2_MOD_NPN:
            if(value->iValue) {
                mods->BJT2type = NPN;
            }
            break;
        case BJT2_MOD_PNP:
            if(value->iValue) {
                mods->BJT2type = PNP;
            }
            break;
        case BJT2_MOD_SUBS:
            mods->BJT2subs = value->iValue;
            mods->BJT2subsGiven = TRUE;
            break;
        case BJT2_MOD_TNOM:
            mods->BJT2tnom = value->rValue+CONSTCtoK;
            mods->BJT2tnomGiven = TRUE;
            break;
        case BJT2_MOD_IS:
            mods->BJT2satCur = value->rValue;
            mods->BJT2satCurGiven = TRUE;
            break;
        case BJT2_MOD_ISS:
            mods->BJT2subSatCur = value->rValue;
            mods->BJT2subSatCurGiven = TRUE;
            break;
        case BJT2_MOD_BF:
            mods->BJT2betaF = value->rValue;
            mods->BJT2betaFGiven = TRUE;
            break;
        case BJT2_MOD_NF:
            mods->BJT2emissionCoeffF = value->rValue;
            mods->BJT2emissionCoeffFGiven = TRUE;
            break;
        case BJT2_MOD_VAF:
            mods->BJT2earlyVoltF = value->rValue;
            mods->BJT2earlyVoltFGiven = TRUE;
            break;
        case BJT2_MOD_IKF:
            mods->BJT2rollOffF = value->rValue;
            mods->BJT2rollOffFGiven = TRUE;
            break;
        case BJT2_MOD_ISE:
            mods->BJT2leakBEcurrent = value->rValue;
            mods->BJT2leakBEcurrentGiven = TRUE;
            break;
        case BJT2_MOD_C2:
            mods->BJT2c2 = value->rValue;
            mods->BJT2c2Given=TRUE;
            break;
        case BJT2_MOD_NE:
            mods->BJT2leakBEemissionCoeff = value->rValue;
            mods->BJT2leakBEemissionCoeffGiven = TRUE;
            break;
        case BJT2_MOD_BR:
            mods->BJT2betaR = value->rValue;
            mods->BJT2betaRGiven = TRUE;
            break;
        case BJT2_MOD_NR:
            mods->BJT2emissionCoeffR = value->rValue;
            mods->BJT2emissionCoeffRGiven = TRUE;
            break;
        case BJT2_MOD_VAR:
            mods->BJT2earlyVoltR = value->rValue;
            mods->BJT2earlyVoltRGiven = TRUE;
            break;
        case BJT2_MOD_IKR:
            mods->BJT2rollOffR = value->rValue;
            mods->BJT2rollOffRGiven = TRUE;
            break;
        case BJT2_MOD_ISC:
            mods->BJT2leakBCcurrent = value->rValue;
            mods->BJT2leakBCcurrentGiven = TRUE;
            break;
        case BJT2_MOD_C4:
            mods->BJT2c4 = value->rValue;
            mods->BJT2c4Given=TRUE;
            break;
        case BJT2_MOD_NC:
            mods->BJT2leakBCemissionCoeff = value->rValue;
            mods->BJT2leakBCemissionCoeffGiven = TRUE;
            break;
        case BJT2_MOD_RB:
            mods->BJT2baseResist = value->rValue;
            mods->BJT2baseResistGiven = TRUE;
            break;
        case BJT2_MOD_IRB:
            mods->BJT2baseCurrentHalfResist = value->rValue;
            mods->BJT2baseCurrentHalfResistGiven = TRUE;
            break;
        case BJT2_MOD_RBM:
            mods->BJT2minBaseResist = value->rValue;
            mods->BJT2minBaseResistGiven = TRUE;
            break;
        case BJT2_MOD_RE:
            mods->BJT2emitterResist = value->rValue;
            mods->BJT2emitterResistGiven = TRUE;
            break;
        case BJT2_MOD_RC:
            mods->BJT2collectorResist = value->rValue;
            mods->BJT2collectorResistGiven = TRUE;
            break;
        case BJT2_MOD_CJE:
            mods->BJT2depletionCapBE = value->rValue;
            mods->BJT2depletionCapBEGiven = TRUE;
            break;
        case BJT2_MOD_VJE:
            mods->BJT2potentialBE = value->rValue;
            mods->BJT2potentialBEGiven = TRUE;
            break;
        case BJT2_MOD_MJE:
            mods->BJT2junctionExpBE = value->rValue;
            mods->BJT2junctionExpBEGiven = TRUE;
            break;
        case BJT2_MOD_TF:
            mods->BJT2transitTimeF = value->rValue;
            mods->BJT2transitTimeFGiven = TRUE;
            break;
        case BJT2_MOD_XTF:
            mods->BJT2transitTimeBiasCoeffF = value->rValue;
            mods->BJT2transitTimeBiasCoeffFGiven = TRUE;
            break;
        case BJT2_MOD_VTF:
            mods->BJT2transitTimeFVBC = value->rValue;
            mods->BJT2transitTimeFVBCGiven = TRUE;
            break;
        case BJT2_MOD_ITF:
            mods->BJT2transitTimeHighCurrentF = value->rValue;
            mods->BJT2transitTimeHighCurrentFGiven = TRUE;
            break;
        case BJT2_MOD_PTF:
            mods->BJT2excessPhase = value->rValue;
            mods->BJT2excessPhaseGiven = TRUE;
            break;
        case BJT2_MOD_CJC:
            mods->BJT2depletionCapBC = value->rValue;
            mods->BJT2depletionCapBCGiven = TRUE;
            break;
        case BJT2_MOD_VJC:
            mods->BJT2potentialBC = value->rValue;
            mods->BJT2potentialBCGiven = TRUE;
            break;
        case BJT2_MOD_MJC:
            mods->BJT2junctionExpBC = value->rValue;
            mods->BJT2junctionExpBCGiven = TRUE;
            break;
        case BJT2_MOD_XCJC:
            mods->BJT2baseFractionBCcap = value->rValue;
            mods->BJT2baseFractionBCcapGiven = TRUE;
            break;
        case BJT2_MOD_TR:
            mods->BJT2transitTimeR = value->rValue;
            mods->BJT2transitTimeRGiven = TRUE;
            break;
        case BJT2_MOD_CJS:
            mods->BJT2capSub = value->rValue;
            mods->BJT2capSubGiven = TRUE;
            break;
        case BJT2_MOD_VJS:
            mods->BJT2potentialSubstrate = value->rValue;
            mods->BJT2potentialSubstrateGiven = TRUE;
            break;
        case BJT2_MOD_MJS:
            mods->BJT2exponentialSubstrate = value->rValue;
            mods->BJT2exponentialSubstrateGiven = TRUE;
            break;
        case BJT2_MOD_XTB:
            mods->BJT2betaExp = value->rValue;
            mods->BJT2betaExpGiven = TRUE;
            break;
        case BJT2_MOD_EG:
            mods->BJT2energyGap = value->rValue;
            mods->BJT2energyGapGiven = TRUE;
            break;
        case BJT2_MOD_XTI:
            mods->BJT2tempExpIS = value->rValue;
            mods->BJT2tempExpISGiven = TRUE;
            break;
        case BJT2_MOD_TRE1:
            mods->BJT2reTempCoeff1 = value->rValue;
            mods->BJT2reTempCoeff1Given = TRUE;
            break;
        case BJT2_MOD_TRE2:
            mods->BJT2reTempCoeff2 = value->rValue;
            mods->BJT2reTempCoeff2Given = TRUE;
            break;
        case BJT2_MOD_TRC1:
            mods->BJT2rcTempCoeff1 = value->rValue;
            mods->BJT2rcTempCoeff1Given = TRUE;
            break;
        case BJT2_MOD_TRC2:
            mods->BJT2rcTempCoeff2 = value->rValue;
            mods->BJT2rcTempCoeff2Given = TRUE;
            break;
        case BJT2_MOD_TRB1:
            mods->BJT2rbTempCoeff1 = value->rValue;
            mods->BJT2rbTempCoeff1Given = TRUE;
            break;
        case BJT2_MOD_TRB2:
            mods->BJT2rbTempCoeff2 = value->rValue;
            mods->BJT2rbTempCoeff2Given = TRUE;
            break;
        case BJT2_MOD_TRBM1:
            mods->BJT2rbmTempCoeff1 = value->rValue;
            mods->BJT2rbmTempCoeff1Given = TRUE;
            break;
        case BJT2_MOD_TRBM2:
            mods->BJT2rbmTempCoeff2 = value->rValue;
            mods->BJT2rbmTempCoeff2Given = TRUE;
            break;
        case BJT2_MOD_FC:
            mods->BJT2depletionCapCoeff = value->rValue;
            mods->BJT2depletionCapCoeffGiven = TRUE;
            break;
	case BJT2_MOD_KF:
	    mods->BJT2fNcoef = value->rValue;
	    mods->BJT2fNcoefGiven = TRUE;
	    break;
	case BJT2_MOD_AF:
	    mods->BJT2fNexp = value->rValue;
	    mods->BJT2fNexpGiven = TRUE;
	    break;
        default:
            return(E_BADPARM);
    }
    return(OK);
}
