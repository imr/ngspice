/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1987 Mathew Lew and Thomas L. Quarles
Modified: Alan Gillespie
**********/
/*
 */

#include "ngspice.h"
#include "const.h"
#include "ifsim.h"
#include "cktdefs.h"
#include "devdefs.h"
#include "bjt2defs.h"
#include "sperror.h"
#include "suffix.h"


/*ARGSUSED*/
int
BJT2mAsk(CKTcircuit *ckt, GENmodel *instPtr, int which, IFvalue *value)
{
    BJT2model *here = (BJT2model*)instPtr;

    switch(which) {
        case BJT2_MOD_TNOM:
            value->rValue = here->BJT2tnom-CONSTCtoK;
            return(OK);
        case BJT2_MOD_IS:
            value->rValue = here->BJT2satCur;
            return(OK);
        case BJT2_MOD_ISS:
            value->rValue = here->BJT2subSatCur;
            return(OK);
        case BJT2_MOD_BF:
            value->rValue = here->BJT2betaF;
            return(OK);
        case BJT2_MOD_NF:
            value->rValue = here->BJT2emissionCoeffF;
            return(OK);
        case BJT2_MOD_VAF:
            value->rValue = here->BJT2earlyVoltF;
            return(OK);
        case BJT2_MOD_IKF:
            value->rValue = here->BJT2rollOffF;
            return(OK);
        case BJT2_MOD_ISE:
            value->rValue = here->BJT2leakBEcurrent;
            return(OK);
        case BJT2_MOD_C2:
            value->rValue = here->BJT2c2;
            return(OK);
        case BJT2_MOD_NE:
            value->rValue = here->BJT2leakBEemissionCoeff;
            return(OK);
        case BJT2_MOD_BR:
            value->rValue = here->BJT2betaR;
            return(OK);
        case BJT2_MOD_NR:
            value->rValue = here->BJT2emissionCoeffR;
            return(OK);
        case BJT2_MOD_VAR:
            value->rValue = here->BJT2earlyVoltR;
            return(OK);
        case BJT2_MOD_IKR:
            value->rValue = here->BJT2rollOffR;
            return(OK);
        case BJT2_MOD_ISC:
            value->rValue = here->BJT2leakBCcurrent;
            return(OK);
        case BJT2_MOD_C4:
            value->rValue = here->BJT2c4;
            return(OK);
        case BJT2_MOD_NC:
            value->rValue = here->BJT2leakBCemissionCoeff;
            return(OK);
        case BJT2_MOD_RB:
            value->rValue = here->BJT2baseResist;
            return(OK);
        case BJT2_MOD_IRB:
            value->rValue = here->BJT2baseCurrentHalfResist;
            return(OK);
        case BJT2_MOD_RBM:
            value->rValue = here->BJT2minBaseResist;
            return(OK);
        case BJT2_MOD_RE:
            value->rValue = here->BJT2emitterResist;
            return(OK);
        case BJT2_MOD_RC:
            value->rValue = here->BJT2collectorResist;
            return(OK);
        case BJT2_MOD_CJE:
            value->rValue = here->BJT2depletionCapBE;
            return(OK);
        case BJT2_MOD_VJE:
            value->rValue = here->BJT2potentialBE;
            return(OK);
        case BJT2_MOD_MJE:
            value->rValue = here->BJT2junctionExpBE;
            return(OK);
        case BJT2_MOD_TF:
            value->rValue = here->BJT2transitTimeF;
            return(OK);
        case BJT2_MOD_XTF:
            value->rValue = here->BJT2transitTimeBiasCoeffF;
            return(OK);
        case BJT2_MOD_VTF:
            value->rValue = here->BJT2transitTimeFVBC;
            return(OK);
        case BJT2_MOD_ITF:
            value->rValue = here->BJT2transitTimeHighCurrentF;
            return(OK);
        case BJT2_MOD_PTF:
            value->rValue = here->BJT2excessPhase;
            return(OK);
        case BJT2_MOD_CJC:
            value->rValue = here->BJT2depletionCapBC;
            return(OK);
        case BJT2_MOD_VJC:
            value->rValue = here->BJT2potentialBC;
            return(OK);
        case BJT2_MOD_MJC:
            value->rValue = here->BJT2junctionExpBC;
            return(OK);
        case BJT2_MOD_XCJC:
            value->rValue = here->BJT2baseFractionBCcap;
            return(OK);
        case BJT2_MOD_TR:
            value->rValue = here->BJT2transitTimeR;
            return(OK);
        case BJT2_MOD_CJS:
            value->rValue = here->BJT2capSub;
            return(OK);
        case BJT2_MOD_VJS:
            value->rValue = here->BJT2potentialSubstrate;
            return(OK);
        case BJT2_MOD_MJS:
            value->rValue = here->BJT2exponentialSubstrate;
            return(OK);
        case BJT2_MOD_XTB:
            value->rValue = here->BJT2betaExp;
            return(OK);
        case BJT2_MOD_EG:
            value->rValue = here->BJT2energyGap;
            return(OK);
        case BJT2_MOD_XTI:
            value->rValue = here->BJT2tempExpIS;
            return(OK);
        case BJT2_MOD_TRE1:
            value->rValue = here->BJT2reTempCoeff1;
            return(OK);
        case BJT2_MOD_TRE2:
            value->rValue = here->BJT2reTempCoeff2;
            return(OK);
        case BJT2_MOD_TRC1:
            value->rValue = here->BJT2rcTempCoeff1;
            return(OK);
        case BJT2_MOD_TRC2:
            value->rValue = here->BJT2rcTempCoeff2;
            return(OK);
        case BJT2_MOD_TRB1:
            value->rValue = here->BJT2rbTempCoeff1;
            return(OK);
        case BJT2_MOD_TRB2:
            value->rValue = here->BJT2rbTempCoeff2;
            return(OK);
        case BJT2_MOD_TRBM1:
            value->rValue = here->BJT2rbmTempCoeff1;
            return(OK);
        case BJT2_MOD_TRBM2:
            value->rValue = here->BJT2rbmTempCoeff2;
            return(OK);
        case BJT2_MOD_FC:
            value->rValue = here->BJT2depletionCapCoeff;
            return(OK);
        case BJT2_MOD_INVEARLYF:
            value->rValue = here->BJT2invEarlyVoltF;
            return(OK);
        case BJT2_MOD_INVEARLYR:
            value->rValue = here->BJT2invEarlyVoltR;
            return(OK);
        case BJT2_MOD_INVROLLOFFF:
            value->rValue = here->BJT2invRollOffF;
            return(OK);
        case BJT2_MOD_INVROLLOFFR:
            value->rValue = here->BJT2invRollOffR;
            return(OK);
        case BJT2_MOD_COLCONDUCT:
            value->rValue = here->BJT2collectorConduct;
            return(OK);
        case BJT2_MOD_EMITTERCONDUCT:
            value->rValue = here->BJT2emitterConduct;
            return(OK);
        case BJT2_MOD_TRANSVBCFACT:
            value->rValue = here->BJT2transitTimeVBCFactor;
            return(OK);
        case BJT2_MOD_EXCESSPHASEFACTOR:
            value->rValue = here->BJT2excessPhaseFactor;
            return(OK);
	case BJT2_MOD_KF:
	    if (here->BJT2fNcoefGiven)
	        value->rValue = here->BJT2fNcoef;
	    else
	        value->rValue = 0.0;
            return(OK);
	case BJT2_MOD_AF:
	    if (here->BJT2fNexpGiven)
	        value->rValue = here->BJT2fNexp;
	    else
	        value->rValue = 0.0;
            return(OK);
	case BJT2_MOD_TYPE:
	    if (here->BJT2type == NPN)
	        value->sValue = "npn";
	    else
	        value->sValue = "pnp";
            return(OK);
	case BJT2_MOD_SUBS:
	    if (here->BJT2subs == LATERAL)
	        value->sValue = "Lateral";
	    else
	        value->sValue = "Vertical";
            return(OK);
        default:
            return(E_BADPARM);
    }
    /* NOTREACHED */
}

