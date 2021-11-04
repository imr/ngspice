/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1987 Mathew Lew and Thomas L. Quarles
**********/
/*
 */

#include "ngspice/ngspice.h"
#include "ngspice/const.h"
#include "ngspice/ifsim.h"
#include "ngspice/cktdefs.h"
#include "ngspice/devdefs.h"
#include "bjtdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


/*ARGSUSED*/
int
BJTmAsk(CKTcircuit *ckt, GENmodel *instPtr, int which, IFvalue *value)
{
    BJTmodel *here = (BJTmodel*)instPtr;

    NG_IGNORE(ckt);

    switch(which) {
        case BJT_MOD_TYPE:
            if (here->BJTtype == NPN)
                value->sValue = "npn";
            else
                value->sValue = "pnp";
            return(OK);
        case BJT_MOD_SUBS:
            if (here->BJTsubs == LATERAL)
                value->sValue = "Lateral";
            else
                value->sValue = "Vertical";
            return(OK);
        case BJT_MOD_TNOM:
            value->rValue = here->BJTtnom-CONSTCtoK;
            return(OK);
        case BJT_MOD_IS:
            value->rValue = here->BJTsatCur;
            return(OK);
        case BJT_MOD_IBE:
            value->rValue = here->BJTBEsatCur;
            return(OK);
        case BJT_MOD_IBC:
            value->rValue = here->BJTBCsatCur;
            return(OK);
        case BJT_MOD_BF:
            value->rValue = here->BJTbetaF;
            return(OK);
        case BJT_MOD_NF:
            value->rValue = here->BJTemissionCoeffF;
            return(OK);
        case BJT_MOD_VAF:
            value->rValue = here->BJTearlyVoltF;
            return(OK);
        case BJT_MOD_IKF:
            value->rValue = here->BJTrollOffF;
            return(OK);
        case BJT_MOD_ISE:
            value->rValue = here->BJTleakBEcurrent;
            return(OK);
        case BJT_MOD_NE:
            value->rValue = here->BJTleakBEemissionCoeff;
            return(OK);
        case BJT_MOD_BR:
            value->rValue = here->BJTbetaR;
            return(OK);
        case BJT_MOD_NR:
            value->rValue = here->BJTemissionCoeffR;
            return(OK);
        case BJT_MOD_VAR:
            value->rValue = here->BJTearlyVoltR;
            return(OK);
        case BJT_MOD_IKR:
            value->rValue = here->BJTrollOffR;
            return(OK);
        case BJT_MOD_ISC:
            value->rValue = here->BJTleakBCcurrent;
            return(OK);
        case BJT_MOD_NC:
            value->rValue = here->BJTleakBCemissionCoeff;
            return(OK);
        case BJT_MOD_RB:
            value->rValue = here->BJTbaseResist;
            return(OK);
        case BJT_MOD_IRB:
            value->rValue = here->BJTbaseCurrentHalfResist;
            return(OK);
        case BJT_MOD_RBM:
            value->rValue = here->BJTminBaseResist;
            return(OK);
        case BJT_MOD_RE:
            value->rValue = here->BJTemitterResist;
            return(OK);
        case BJT_MOD_RC:
            value->rValue = here->BJTcollectorResist;
            return(OK);
        case BJT_MOD_CJE:
            value->rValue = here->BJTdepletionCapBE;
            return(OK);
        case BJT_MOD_VJE:
            value->rValue = here->BJTpotentialBE;
            return(OK);
        case BJT_MOD_MJE:
            value->rValue = here->BJTjunctionExpBE;
            return(OK);
        case BJT_MOD_TF:
            value->rValue = here->BJTtransitTimeF;
            return(OK);
        case BJT_MOD_XTF:
            value->rValue = here->BJTtransitTimeBiasCoeffF;
            return(OK);
        case BJT_MOD_VTF:
            value->rValue = here->BJTtransitTimeFVBC;
            return(OK);
        case BJT_MOD_ITF:
            value->rValue = here->BJTtransitTimeHighCurrentF;
            return(OK);
        case BJT_MOD_PTF:
            value->rValue = here->BJTexcessPhase;
            return(OK);
        case BJT_MOD_CJC:
            value->rValue = here->BJTdepletionCapBC;
            return(OK);
        case BJT_MOD_VJC:
            value->rValue = here->BJTpotentialBC;
            return(OK);
        case BJT_MOD_MJC:
            value->rValue = here->BJTjunctionExpBC;
            return(OK);
        case BJT_MOD_XCJC:
            value->rValue = here->BJTbaseFractionBCcap;
            return(OK);
        case BJT_MOD_TR:
            value->rValue = here->BJTtransitTimeR;
            return(OK);
        case BJT_MOD_CJS:
            value->rValue = here->BJTcapSub;
            return(OK);
        case BJT_MOD_VJS:
            value->rValue = here->BJTpotentialSubstrate;
            return(OK);
        case BJT_MOD_MJS:
            value->rValue = here->BJTexponentialSubstrate;
            return(OK);
        case BJT_MOD_XTB:
            value->rValue = here->BJTbetaExp;
            return(OK);
        case BJT_MOD_EG:
            value->rValue = here->BJTenergyGap;
            return(OK);
        case BJT_MOD_XTI:
            value->rValue = here->BJTtempExpIS;
            return(OK);
        case BJT_MOD_FC:
            value->rValue = here->BJTdepletionCapCoeff;
            return(OK);
        case BJT_MOD_KF:
            if (here->BJTfNcoefGiven)
                value->rValue = here->BJTfNcoef;
            else
                value->rValue = 0.0;
            return(OK);
        case BJT_MOD_AF:
            if (here->BJTfNexpGiven)
                value->rValue = here->BJTfNexp;
            else
                value->rValue = 0.0;
            return(OK);
        case BJT_MOD_INVEARLYF:
            value->rValue = here->BJTinvEarlyVoltF;
            return(OK);
        case BJT_MOD_INVEARLYR:
            value->rValue = here->BJTinvEarlyVoltR;
            return(OK);
        case BJT_MOD_INVROLLOFFF:
            value->rValue = here->BJTinvRollOffF;
            return(OK);
        case BJT_MOD_INVROLLOFFR:
            value->rValue = here->BJTinvRollOffR;
            return(OK);
        case BJT_MOD_COLCONDUCT:
            value->rValue = here->BJTcollectorConduct;
            return(OK);
        case BJT_MOD_EMITTERCONDUCT:
            value->rValue = here->BJTemitterConduct;
            return(OK);
        case BJT_MOD_TRANSVBCFACT:
            value->rValue = here->BJTtransitTimeVBCFactor;
            return(OK);
        case BJT_MOD_EXCESSPHASEFACTOR:
            value->rValue = here->BJTexcessPhaseFactor;
            return(OK);
        case BJT_MOD_ISS:
            value->rValue = here->BJTsubSatCur;
            return(OK);
        case BJT_MOD_NS:
            value->rValue = here->BJTemissionCoeffS;
            return(OK);
        case BJT_MOD_RCO:
            value->rValue = here->BJTintCollResist;
            return(OK);
        case BJT_MOD_VO:
            value->rValue = here->BJTepiSatVoltage;
            return(OK);
        case BJT_MOD_GAMMA:
            value->rValue = here->BJTepiDoping;
            return(OK);
        case BJT_MOD_QCO:
            value->rValue = here->BJTepiCharge;
            return(OK);
        case BJT_MOD_TLEV:
            value->iValue = here->BJTtlev;
            return(OK);
        case BJT_MOD_TLEVC:
            value->iValue = here->BJTtlevc;
            return(OK);
        case BJT_MOD_TBF1:
            value->rValue = here->BJTtbf1;
            return(OK);
        case BJT_MOD_TBF2:
            value->rValue = here->BJTtbf2;
            return(OK);
        case BJT_MOD_TBR1:
            value->rValue = here->BJTtbr1;
            return(OK);
        case BJT_MOD_TBR2:
            value->rValue = here->BJTtbr2;
            return(OK);
        case BJT_MOD_TIKF1:
            value->rValue = here->BJTtikf1;
            return(OK);
        case BJT_MOD_TIKF2:
            value->rValue = here->BJTtikf2;
            return(OK);
        case BJT_MOD_TIKR1:
            value->rValue = here->BJTtikr1;
            return(OK);
        case BJT_MOD_TIKR2:
            value->rValue = here->BJTtikr2;
            return(OK);
        case BJT_MOD_TIRB1:
            value->rValue = here->BJTtirb1;
            return(OK);
        case BJT_MOD_TIRB2:
            value->rValue = here->BJTtirb2;
            return(OK);
        case BJT_MOD_TNC1:
            value->rValue = here->BJTtnc1;
            return(OK);
        case BJT_MOD_TNC2:
            value->rValue = here->BJTtnc2;
            return(OK);
        case BJT_MOD_TNE1:
            value->rValue = here->BJTtne1;
            return(OK);
        case BJT_MOD_TNE2:
            value->rValue = here->BJTtne2;
            return(OK);
        case BJT_MOD_TNF1:
            value->rValue = here->BJTtnf1;
            return(OK);
        case BJT_MOD_TNF2:
            value->rValue = here->BJTtnf2;
            return(OK);
        case BJT_MOD_TNR1:
            value->rValue = here->BJTtnr1;
            return(OK);
        case BJT_MOD_TNR2:
            value->rValue = here->BJTtnr2;
            return(OK);
        case BJT_MOD_TRB1:
            value->rValue = here->BJTtrb1;
            return(OK);
        case BJT_MOD_TRB2:
            value->rValue = here->BJTtrb2;
            return(OK);
        case BJT_MOD_TRC1:
            value->rValue = here->BJTtrc1;
            return(OK);
        case BJT_MOD_TRC2:
            value->rValue = here->BJTtrc2;
            return(OK);
        case BJT_MOD_TRE1:
            value->rValue = here->BJTtre1;
            return(OK);
        case BJT_MOD_TRE2:
            value->rValue = here->BJTtre2;
            return(OK);
        case BJT_MOD_TRM1:
            value->rValue = here->BJTtrm1;
            return(OK);
        case BJT_MOD_TRM2:
            value->rValue = here->BJTtrm2;
            return(OK);
        case BJT_MOD_TVAF1:
            value->rValue = here->BJTtvaf1;
            return(OK);
        case BJT_MOD_TVAF2:
            value->rValue = here->BJTtvaf2;
            return(OK);
        case BJT_MOD_TVAR1:
            value->rValue = here->BJTtvar1;
            return(OK);
        case BJT_MOD_TVAR2:
            value->rValue = here->BJTtvar2;
            return(OK);
        case BJT_MOD_CTC:
            value->rValue = here->BJTctc;
            return(OK);
        case BJT_MOD_CTE:
            value->rValue = here->BJTcte;
            return(OK);
        case BJT_MOD_CTS:
            value->rValue = here->BJTcts;
            return(OK);
        case BJT_MOD_TVJE:
            value->rValue = here->BJTtvje;
            return(OK);
        case BJT_MOD_TVJC:
            value->rValue = here->BJTtvjc;
            return(OK);
        case BJT_MOD_TVJS:
            value->rValue = here->BJTtvjs;
            return(OK);
        case BJT_MOD_TITF1:
            value->rValue = here->BJTtitf1;
            return(OK);
        case BJT_MOD_TITF2:
            value->rValue = here->BJTtitf2;
            return(OK);
        case BJT_MOD_TTF1:
            value->rValue = here->BJTttf1;
            return(OK);
        case BJT_MOD_TTF2:
            value->rValue = here->BJTttf2;
            return(OK);
        case BJT_MOD_TTR1:
            value->rValue = here->BJTttr1;
            return(OK);
        case BJT_MOD_TTR2:
            value->rValue = here->BJTttr2;
            return(OK);
        case BJT_MOD_TMJE1:
            value->rValue = here->BJTtmje1;
            return(OK);
        case BJT_MOD_TMJE2:
            value->rValue = here->BJTtmje2;
            return(OK);
        case BJT_MOD_TMJC1:
            value->rValue = here->BJTtmjc1;
            return(OK);
        case BJT_MOD_TMJC2:
            value->rValue = here->BJTtmjc2;
            return(OK);
        case BJT_MOD_TMJS1:
            value->rValue = here->BJTtmjs1;
            return(OK);
        case BJT_MOD_TMJS2:
            value->rValue = here->BJTtmjs2;
            return(OK);
        case BJT_MOD_TNS1:
            value->rValue = here->BJTtns1;
            return(OK);
        case BJT_MOD_TNS2:
            value->rValue = here->BJTtns2;
            return(OK);
        case BJT_MOD_NKF:
            value->rValue = here->BJTnkf;
            return(OK);
        case BJT_MOD_TIS1:
            value->rValue = here->BJTtis1;
            return(OK);
        case BJT_MOD_TIS2:
            value->rValue = here->BJTtis2;
            return(OK);
        case BJT_MOD_TISE1:
            value->rValue = here->BJTtise1;
            return(OK);
        case BJT_MOD_TISE2:
            value->rValue = here->BJTtise2;
            return(OK);
        case BJT_MOD_TISC1:
            value->rValue = here->BJTtisc1;
            return(OK);
        case BJT_MOD_TISC2:
            value->rValue = here->BJTtisc2;
            return(OK);
        case BJT_MOD_TISS1:
            value->rValue = here->BJTtiss1;
            return(OK);
        case BJT_MOD_TISS2:
            value->rValue = here->BJTtiss2;
            return(OK);
        case BJT_MOD_QUASIMOD:
            value->iValue = here->BJTquasimod;
            return(OK);
        case BJT_MOD_EGQS:
            value->rValue = here->BJTenergyGapQS;
            return(OK);
        case BJT_MOD_XRCI:
            value->rValue = here->BJTtempExpRCI;
            return(OK);
        case BJT_MOD_XD:
            value->rValue = here->BJTtempExpVO;
            return(OK);
        case BJT_MOD_VBE_MAX:
            value->rValue = here->BJTvbeMax;
            return(OK);
        case BJT_MOD_VBC_MAX:
            value->rValue = here->BJTvbcMax;
            return(OK);
        case BJT_MOD_VCE_MAX:
            value->rValue = here->BJTvceMax;
            return(OK);
        case BJT_MOD_IC_MAX:
            value->rValue = here->BJTicMax;
            return(OK);
        case BJT_MOD_IB_MAX:
            value->rValue = here->BJTibMax;
            return(OK);
        case BJT_MOD_PD_MAX:
            value->rValue = here->BJTpdMax;
            return(OK);
        case BJT_MOD_RTH0:
            value->rValue = here->BJTrth0;
            return(OK);
        case BJT_MOD_TE_MAX:
            value->rValue = here->BJTteMax;
            return(OK);
        default:
            return(E_BADPARM);
    }
    /* NOTREACHED */
}

