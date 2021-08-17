/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified by Paolo Nenzi 2003 and Dietmar Warning 2012
**********/
/*
 */

#include "ngspice/ngspice.h"
#include "ngspice/const.h"
#include "ngspice/cktdefs.h"
#include "ngspice/ifsim.h"
#include "diodefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


/* ARGSUSED */
int
DIOmAsk (CKTcircuit *ckt, GENmodel *inModel, int which, IFvalue *value)
{
    DIOmodel *model = (DIOmodel*)inModel;

    NG_IGNORE(ckt);

    switch (which) {
        case DIO_MOD_LEVEL:
            value->iValue = model->DIOlevel;
            return (OK);
        case DIO_MOD_IS:
            value->rValue = model->DIOsatCur;
            if (value->rValue < ckt->CKTepsmin)
                value->rValue = ckt->CKTepsmin;
            return(OK);
        case DIO_MOD_JSW:
            value->rValue = model->DIOsatSWCur;
            return(OK);

        case DIO_MOD_TNOM:
            value->rValue = model->DIOnomTemp-CONSTCtoK;
            return(OK);
        case DIO_MOD_RS:
            value->rValue = model->DIOresist;
            return(OK);
        case DIO_MOD_TRS:
            value->rValue = model->DIOresistTemp1;
            return(OK);
        case DIO_MOD_TRS2:
            value->rValue = model->DIOresistTemp2;
            return(OK);
        case DIO_MOD_N:
            value->rValue = model->DIOemissionCoeff;
            return(OK);
        case DIO_MOD_NS:
            value->rValue = model->DIOswEmissionCoeff;
            return(OK);
        case DIO_MOD_TT:
            value->rValue = model->DIOtransitTime;
            return(OK);
        case DIO_MOD_TTT1:
            value->rValue = model->DIOtranTimeTemp1;
            return(OK);
        case DIO_MOD_TTT2:
            value->rValue = model->DIOtranTimeTemp2;
            return(OK);
        case DIO_MOD_CJO:
            value->rValue = model->DIOjunctionCap;
            return(OK);
        case DIO_MOD_VJ:
            value->rValue = model->DIOjunctionPot;
            return(OK);
        case DIO_MOD_M:
            value->rValue = model->DIOgradingCoeff;
            return(OK);
        case DIO_MOD_TM1:
            value->rValue = model->DIOgradCoeffTemp1;
            return(OK);
        case DIO_MOD_TM2:
            value->rValue = model->DIOgradCoeffTemp2;
            return(OK);
        case DIO_MOD_CJSW:
            value->rValue = model->DIOjunctionSWCap;
            return(OK);
        case DIO_MOD_VJSW:
            value->rValue = model->DIOjunctionSWPot;
            return(OK);
        case DIO_MOD_MJSW:
            value->rValue = model->DIOgradingSWCoeff;
            return(OK);
        case DIO_MOD_IKF:
            value->rValue = model->DIOforwardKneeCurrent;
            return(OK);
        case DIO_MOD_IKR:
            value->rValue = model->DIOreverseKneeCurrent;
            return(OK);
        case DIO_MOD_NBV:
            value->rValue = model->DIObrkdEmissionCoeff;
            return(OK);

        case DIO_MOD_TLEV:
            value->iValue = model->DIOtlev;
            return (OK);
        case DIO_MOD_TLEVC:
            value->iValue = model->DIOtlevc;
            return (OK);
        case DIO_MOD_EG:
            value->rValue = model->DIOactivationEnergy;
            return (OK);
        case DIO_MOD_XTI:
            value->rValue = model->DIOsaturationCurrentExp;
            return(OK);
        case DIO_MOD_CTA:
            value->rValue = model->DIOcta;
            return(OK);
        case DIO_MOD_CTP:
            value->rValue = model->DIOctp;
            return(OK);
        case DIO_MOD_TPB:
            value->rValue = model->DIOtpb;
            return(OK);
        case DIO_MOD_TPHP:
            value->rValue = model->DIOtphp;
            return(OK);
        case DIO_MOD_FC:
            value->rValue = model->DIOdepletionCapCoeff;
            return(OK);
        case DIO_MOD_FCS:
            value->rValue = model->DIOdepletionSWcapCoeff;
            return(OK);
        case DIO_MOD_KF:
            value->rValue = model->DIOfNcoef;
            return(OK);
        case DIO_MOD_AF:
            value->rValue = model->DIOfNexp;
            return(OK);
        case DIO_MOD_BV:
            value->rValue = model->DIObreakdownVoltage;
            return(OK);
        case DIO_MOD_IBV:
            value->rValue = model->DIObreakdownCurrent;
            return(OK);
        case DIO_MOD_TCV:
            value->rValue = model->DIOtcv;
            return(OK);
        case DIO_MOD_AREA:
            value->rValue = model->DIOarea;
            return(OK);
        case DIO_MOD_PJ:
            value->rValue = model->DIOpj;
            return(OK);
        case DIO_MOD_COND:
            value->rValue = model->DIOconductance;
            return(OK);
        case DIO_MOD_JTUN:
            value->rValue = model->DIOtunSatCur;
            return(OK);
        case DIO_MOD_JTUNSW:
            value->rValue = model->DIOtunSatSWCur;
            return(OK);
        case DIO_MOD_NTUN:
            value->rValue = model->DIOtunEmissionCoeff;
            return(OK);
        case DIO_MOD_XTITUN:
            value->rValue = model->DIOtunSaturationCurrentExp;
            return(OK);
        case DIO_MOD_KEG:
            value->rValue = model->DIOtunEGcorrectionFactor;
            return(OK);
        case DIO_MOD_FV_MAX:
            value->rValue = model->DIOfv_max;
            return(OK);
        case DIO_MOD_BV_MAX:
            value->rValue = model->DIObv_max;
            return(OK);
        case DIO_MOD_ID_MAX:
            value->rValue = model->DIOid_max;
            return(OK);
        case DIO_MOD_PD_MAX:
            value->rValue = model->DIOpd_max;
            return(OK);
        case DIO_MOD_TE_MAX:
            value->rValue = model->DIOte_max;
            return(OK);
        case DIO_MOD_ISR:
            value->rValue = model->DIOrecSatCur;
            return(OK);
        case DIO_MOD_NR:
            value->rValue = model->DIOrecEmissionCoeff;
            return(OK);
        case DIO_MOD_RTH0:
            value->rValue = model->DIOrth0; 
            return(OK);
        case DIO_MOD_CTH0:
            value->rValue = model->DIOcth0; 
            return(OK);

        case DIO_MOD_LM:
            value->rValue = model->DIOlengthMetal;
            return(OK);
        case DIO_MOD_LP:
            value->rValue = model->DIOlengthPoly;
            return(OK);
        case DIO_MOD_WM:
            value->rValue = model->DIOwidthMetal;
            return(OK);
        case DIO_MOD_WP:
            value->rValue = model->DIOwidthPoly;
            return(OK);
        case DIO_MOD_XOM:
            value->rValue = model->DIOmetalOxideThick;
            return(OK);
        case DIO_MOD_XOI:
            value->rValue = model->DIOpolyOxideThick;
            return(OK);
        case DIO_MOD_XM:
            value->rValue = model->DIOmetalMaskOffset;
            return(OK);
        case DIO_MOD_XP:
            value->rValue = model->DIOpolyMaskOffset;
            return(OK);

        default:
            return(E_BADPARM);
        }
}

