/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified by Paolo Nenzi 2003 and Dietmar Warning 2012
**********/
/*
 */

#include "ngspice/ngspice.h"
#include "ngspice/const.h"
#include "ngspice/ifsim.h"
#include "diodefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
DIOmParam(int param, IFvalue *value, GENmodel *inModel)
{
    DIOmodel *model = (DIOmodel*)inModel;
    switch(param) {
        case DIO_MOD_LEVEL:
            model->DIOlevel = value->iValue;
            model->DIOlevelGiven = TRUE;
            break;
        case DIO_MOD_IS:
            model->DIOsatCur = value->rValue;
            model->DIOsatCurGiven = TRUE;
            break;
        case DIO_MOD_JSW:
            model->DIOsatSWCur = value->rValue;
            model->DIOsatSWCurGiven = TRUE;
            break;

        case DIO_MOD_TNOM:
            model->DIOnomTemp = value->rValue+CONSTCtoK;
            model->DIOnomTempGiven = TRUE;
            break;
        case DIO_MOD_RS:
            model->DIOresist = value->rValue;
            model->DIOresistGiven = TRUE;
            break;
        case DIO_MOD_TRS:
            model->DIOresistTemp1 = value->rValue;
            model->DIOresistTemp1Given = TRUE;
            break;
        case DIO_MOD_TRS2:
            model->DIOresistTemp2 = value->rValue;
            model->DIOresistTemp2Given = TRUE;
            break;
        case DIO_MOD_N:
            model->DIOemissionCoeff = value->rValue;
            model->DIOemissionCoeffGiven = TRUE;
            break;
        case DIO_MOD_NS:
            model->DIOswEmissionCoeff = value->rValue;
            model->DIOswEmissionCoeffGiven = TRUE;
            break;
        case DIO_MOD_TT:
            model->DIOtransitTime = value->rValue;
            model->DIOtransitTimeGiven = TRUE;
            break;
        case DIO_MOD_TTT1:
            model->DIOtranTimeTemp1 = value->rValue;
            model->DIOtranTimeTemp1Given = TRUE;
            break;
        case DIO_MOD_TTT2:
            model->DIOtranTimeTemp2 = value->rValue;
            model->DIOtranTimeTemp2Given = TRUE;
            break;
        case DIO_MOD_CJO:
            model->DIOjunctionCap = value->rValue;
            model->DIOjunctionCapGiven = TRUE;
            break;
        case DIO_MOD_VJ:
            model->DIOjunctionPot = value->rValue;
            model->DIOjunctionPotGiven = TRUE;
            break;
        case DIO_MOD_M:
            model->DIOgradingCoeff = value->rValue;
            model->DIOgradingCoeffGiven = TRUE;
            break;
        case DIO_MOD_TM1:
            model->DIOgradCoeffTemp1 = value->rValue;
            model->DIOgradCoeffTemp1Given = TRUE;
            break;
        case DIO_MOD_TM2:
            model->DIOgradCoeffTemp2 = value->rValue;
            model->DIOgradCoeffTemp2Given = TRUE;
            break;
        case DIO_MOD_CJSW:
            model->DIOjunctionSWCap = value->rValue;
            model->DIOjunctionSWCapGiven = TRUE;
            break;
        case DIO_MOD_VJSW:
            model->DIOjunctionSWPot = value->rValue;
            model->DIOjunctionSWPotGiven = TRUE;
            break;
        case DIO_MOD_MJSW:
            model->DIOgradingSWCoeff = value->rValue;
            model->DIOgradingSWCoeffGiven = TRUE;
            break;
        case DIO_MOD_IKF:
            model->DIOforwardKneeCurrent = value->rValue;
            model->DIOforwardKneeCurrentGiven = TRUE;
            break;
        case DIO_MOD_IKR:
            model->DIOreverseKneeCurrent = value->rValue;
            model->DIOreverseKneeCurrentGiven = TRUE;
            break;
        case DIO_MOD_NBV:
            model->DIObrkdEmissionCoeff = value->rValue;
            model->DIObrkdEmissionCoeffGiven = TRUE;
            break;
        case DIO_MOD_AREA:
            model->DIOarea = value->rValue;
            model->DIOareaGiven = TRUE;
            break;
        case DIO_MOD_PJ:
            model->DIOpj = value->rValue;
            model->DIOpjGiven = TRUE;
            break;

        case DIO_MOD_TLEV:
            model->DIOtlev = value->iValue;
            model->DIOtlevGiven = TRUE;
            break;
        case DIO_MOD_TLEVC:
            model->DIOtlevc = value->iValue;
            model->DIOtlevcGiven = TRUE;
            break;
        case DIO_MOD_EG:
            model->DIOactivationEnergy = value->rValue;
            model->DIOactivationEnergyGiven = TRUE;
            break;
        case DIO_MOD_XTI:
            model->DIOsaturationCurrentExp = value->rValue;
            model->DIOsaturationCurrentExpGiven = TRUE;
            break;
        case DIO_MOD_CTA:
            model->DIOcta = value->rValue;
            model->DIOctaGiven = TRUE;
            break;
        case DIO_MOD_CTP:
            model->DIOctp = value->rValue;
            model->DIOctpGiven = TRUE;
            break;
        case DIO_MOD_TPB:
            model->DIOtpb = value->rValue;
            model->DIOtpbGiven = TRUE;
            break;
        case DIO_MOD_TPHP:
            model->DIOtphp = value->rValue;
            model->DIOtphpGiven = TRUE;
            break;
        case DIO_MOD_FC:
            model->DIOdepletionCapCoeff = value->rValue;
            model->DIOdepletionCapCoeffGiven = TRUE;
            break;
        case DIO_MOD_FCS:
            model->DIOdepletionSWcapCoeff = value->rValue;
            model->DIOdepletionSWcapCoeffGiven = TRUE;
            break;
        case DIO_MOD_BV:
            model->DIObreakdownVoltage = value->rValue;
            model->DIObreakdownVoltageGiven = TRUE;
            break;
        case DIO_MOD_IBV:
            model->DIObreakdownCurrent = value->rValue;
            model->DIObreakdownCurrentGiven = TRUE;
            break;
        case DIO_MOD_TCV:
            model->DIOtcv = value->rValue;
            model->DIOtcvGiven = TRUE;
            break;
        case DIO_MOD_KF:
            model->DIOfNcoef = value->rValue;
            model->DIOfNcoefGiven = TRUE;
            break;
        case DIO_MOD_AF:
            model->DIOfNexp = value->rValue;
            model->DIOfNexpGiven = TRUE;
            break;
        case DIO_MOD_JTUN:
            model->DIOtunSatCur = value->rValue;
            model->DIOtunSatCurGiven = TRUE;
            break;
        case DIO_MOD_JTUNSW:
            model->DIOtunSatSWCur = value->rValue;
            model->DIOtunSatSWCurGiven = TRUE;
            break;
        case DIO_MOD_NTUN:
            model->DIOtunEmissionCoeff = value->rValue;
            model->DIOtunEmissionCoeffGiven = TRUE;
            break;
        case DIO_MOD_XTITUN:
            model->DIOtunSaturationCurrentExp = value->rValue;
            model->DIOtunSaturationCurrentExpGiven = TRUE;
            break;
        case DIO_MOD_KEG:
            model->DIOtunEGcorrectionFactor = value->rValue;
            model->DIOtunEGcorrectionFactorGiven = TRUE;
            break;
        case DIO_MOD_FV_MAX:
            model->DIOfv_max = value->rValue;
            model->DIOfv_maxGiven = TRUE;
            break;
        case DIO_MOD_BV_MAX:
            model->DIObv_max = value->rValue;
            model->DIObv_maxGiven = TRUE;
            break;
        case DIO_MOD_ID_MAX:
            model->DIOid_max = value->rValue;
            model->DIOid_maxGiven = TRUE;
            break;
        case DIO_MOD_PD_MAX:
            model->DIOpd_max = value->rValue;
            model->DIOpd_maxGiven = TRUE;
            break;
        case DIO_MOD_TE_MAX:
            model->DIOte_max = value->rValue;
            model->DIOte_maxGiven = TRUE;
            break;
        case DIO_MOD_ISR:
            model->DIOrecSatCur = value->rValue;
            model->DIOrecSatCurGiven = TRUE;
            break;
        case DIO_MOD_NR:
            model->DIOrecEmissionCoeff = value->rValue;
            model->DIOrecEmissionCoeffGiven = TRUE;
            break;
        case  DIO_MOD_RTH0:
            model->DIOrth0 = value->rValue;
            model->DIOrth0Given = TRUE;
            break;
        case  DIO_MOD_CTH0:
            model->DIOcth0 = value->rValue;
            model->DIOcth0Given = TRUE;
            break;

        case DIO_MOD_LM:
            model->DIOlengthMetal = value->rValue;
            model->DIOlengthMetalGiven = TRUE;
            break;
        case DIO_MOD_LP:
            model->DIOlengthPoly = value->rValue;
            model->DIOlengthPolyGiven = TRUE;
            break;
        case DIO_MOD_WM:
            model->DIOwidthMetal = value->rValue;
            model->DIOwidthMetalGiven = TRUE;
            break;
        case DIO_MOD_WP:
            model->DIOwidthPoly = value->rValue;
            model->DIOwidthPolyGiven = TRUE;
            break;
        case DIO_MOD_XOM:
            model->DIOmetalOxideThick = value->rValue * 1e-10; /* m */
            model->DIOmetalOxideThickGiven = TRUE;
            break;
        case DIO_MOD_XOI:
            model->DIOpolyOxideThick = value->rValue * 1e-10; /* m */
            model->DIOpolyOxideThickGiven = TRUE;
            break;
        case DIO_MOD_XM:
            model->DIOmetalMaskOffset = value->rValue;
            model->DIOmetalMaskOffsetGiven = TRUE;
            break;
        case DIO_MOD_XP:
            model->DIOpolyMaskOffset = value->rValue;
            model->DIOpolyMaskOffsetGiven = TRUE;
            break;

        case DIO_MOD_D:
            /* no action - we already know we are a diode, but this */
            /* makes life easier for spice-2 like parsers */
            break;
        default:
            return(E_BADPARM);
    }
    return(OK);
}
