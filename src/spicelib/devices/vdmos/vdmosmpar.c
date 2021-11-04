/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
VDMOS: 2018 Holger Vogt, 2020 Dietmar Warning
**********/

#include "ngspice/ngspice.h"
#include "ngspice/const.h"
#include "ngspice/ifsim.h"
#include "vdmosdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int
VDMOSmParam(int param, IFvalue *value, GENmodel *inModel)
{
    VDMOSmodel *model = (VDMOSmodel *)inModel;
    switch(param) {
        case VDMOS_MOD_TNOM:
            model->VDMOStnom = value->rValue + CONSTCtoK;
            model->VDMOStnomGiven = TRUE;
            break;
        case VDMOS_MOD_VTH:
            model->VDMOSvth0 = value->rValue;
            model->VDMOSvth0Given = TRUE;
            break;
        case VDMOS_MOD_KP:
            model->VDMOStransconductance = value->rValue;
            model->VDMOStransconductanceGiven = TRUE;
            break;
        case VDMOS_MOD_PHI:
            model->VDMOSphi = value->rValue;
            model->VDMOSphiGiven = TRUE;
            break;
        case VDMOS_MOD_LAMBDA:
            model->VDMOSlambda = value->rValue;
            model->VDMOSlambdaGiven = TRUE;
            break;
        case VDMOS_MOD_THETA:
            model->VDMOStheta = value->rValue;
            model->VDMOSthetaGiven = TRUE;
            break;
        case VDMOS_MOD_RD:
            model->VDMOSdrainResistance = value->rValue;
            model->VDMOSdrainResistanceGiven = TRUE;
            break;
        case VDMOS_MOD_RS:
            model->VDMOSsourceResistance = value->rValue;
            model->VDMOSsourceResistanceGiven = TRUE;
            break;
        case VDMOS_MOD_RG:
            model->VDMOSgateResistance = value->rValue;
            model->VDMOSgateResistanceGiven = TRUE;
            break;
        case VDMOS_MOD_RQ:
            model->VDMOSqsResistance = value->rValue;
            model->VDMOSqsResistanceGiven = TRUE;
            break;
        case VDMOS_MOD_VQ:
            model->VDMOSqsVoltage = value->rValue;
            model->VDMOSqsVoltageGiven = TRUE;
            break;
        case VDIO_MOD_RB:
            model->VDIOresistance = value->rValue;
            model->VDIOresistanceGiven = TRUE;
            break;
        case VDIO_MOD_IS:
            model->VDIOjctSatCur = value->rValue;
            model->VDIOjctSatCurGiven = TRUE;
            break;
        case VDIO_MOD_VJ:
            model->VDIOjunctionPot = value->rValue;
            model->VDIOjunctionPotGiven = TRUE;
            break;
        case VDIO_MOD_CJ:
            model->VDIOjunctionCap = value->rValue;
            model->VDIOjunctionCapGiven = TRUE;
            break;
        case VDIO_MOD_MJ:
            model->VDIOgradCoeff = value->rValue;
            model->VDIOgradCoeffGiven = TRUE;
            model->VDIOgradCoeffTemp1 = 0;
            model->VDIOgradCoeffTemp2 = 0;
            break;
        case VDIO_MOD_FC:
            model->VDIOdepletionCapCoeff = value->rValue;
            model->VDIOdepletionCapCoeffGiven = TRUE;
            break;
        case VDMOS_MOD_NMOS:
            if(value->iValue) {
                model->VDMOStype = 1;
                model->VDMOStypeGiven = TRUE;
            }
            break;
        case VDMOS_MOD_PMOS:
            if(value->iValue) {
                model->VDMOStype = -1;
                model->VDMOStypeGiven = TRUE;
            }
            break;
        case VDMOS_MOD_KF:
            model->VDMOSfNcoef = value->rValue;
            model->VDMOSfNcoefGiven = TRUE;
            break;
        case VDMOS_MOD_AF:
            model->VDMOSfNexp = value->rValue;
            model->VDMOSfNexpGiven = TRUE;
            break;
        case VDMOS_MOD_DMOS:
            if (value->iValue) {
                model->VDMOStype = 1;
                model->VDMOStypeGiven = TRUE;
            }
            break;
        case VDMOS_MOD_CGDMIN:
            model->VDMOScgdmin = value->rValue;
            model->VDMOScgdminGiven = TRUE;
            break;
        case VDMOS_MOD_CGDMAX:
            model->VDMOScgdmax = value->rValue;
            model->VDMOScgdmaxGiven = TRUE;
            break;
        case VDMOS_MOD_A:
            model->VDMOSa = value->rValue;
            model->VDMOSaGiven = TRUE;
            break;
        case VDMOS_MOD_CGS:
            model->VDMOScgs = value->rValue;
            model->VDMOScgsGiven = TRUE;
            break;
        case VDMOS_MOD_MTRIODE:
            model->VDMOSmtr = value->rValue;
            model->VDMOSmtrGiven = TRUE;
            break;
        case VDMOS_MOD_SUBSHIFT:
            model->VDMOSsubshift = value->rValue;
            model->VDMOSsubshiftGiven = TRUE;
            break;
        case VDMOS_MOD_KSUBTHRES:
            model->VDMOSksubthres = value->rValue;
            model->VDMOSksubthresGiven = TRUE;
            break;
        case VDIO_MOD_BV:
            model->VDIObv = value->rValue;
            model->VDIObvGiven = TRUE;
            break;
        case VDIO_MOD_IBV:
            model->VDIOibv = value->rValue;
            model->VDIOibvGiven = TRUE;
            break;
        case VDIO_MOD_NBV:
            model->VDIObrkdEmissionCoeff = value->rValue;
            model->VDIObrkdEmissionCoeffGiven = TRUE;
            break;
        case VDMOS_MOD_RDS:
            model->VDMOSrds = value->rValue;
            model->VDMOSrdsGiven = TRUE;
            break;
        case VDIO_MOD_N:
            model->VDIOn = value->rValue;
            model->VDIOnGiven = TRUE;
            break;
        case VDIO_MOD_TT:
            model->VDIOtransitTime = value->rValue;
            model->VDIOtransitTimeGiven = TRUE;
            model->VDIOtranTimeTemp1 = 0;
            model->VDIOtranTimeTemp2 = 0;
            break;
        case VDIO_MOD_EG:
            model->VDIOeg = value->rValue;
            model->VDIOegGiven = TRUE;
            break;
        case VDIO_MOD_XTI:
            model->VDIOxti = value->rValue;
            model->VDIOxtiGiven = TRUE;
            break;
        case VDMOS_MOD_TCVTH:
            model->VDMOStcvth = value->rValue;
            model->VDMOStcvthGiven = TRUE;
            break;
        case  VDMOS_MOD_RTHJC:
            model->VDMOSrthjc = value->rValue;
            model->VDMOSrthjcGiven = TRUE;
            break;
        case  VDMOS_MOD_RTHCA:
            model->VDMOSrthca = value->rValue;
            model->VDMOSrthcaGiven = TRUE;
            break;
        case  VDMOS_MOD_CTHJ:
            model->VDMOScthj = value->rValue;
            model->VDMOScthjGiven = TRUE;
            break;
        case  VDMOS_MOD_MU:
            model->VDMOSmu = value->rValue;
            model->VDMOSmuGiven = TRUE;
            break;
        case  VDMOS_MOD_TEXP0:
            model->VDMOStexp0 = value->rValue;
            model->VDMOStexp0Given = TRUE;
            break;
        case  VDMOS_MOD_TEXP1:
            model->VDMOStexp1 = value->rValue;
            model->VDMOStexp1Given = TRUE;
            break;
        case  VDMOS_MOD_TRD1:
            model->VDMOStrd1 = value->rValue;
            model->VDMOStrd1Given = TRUE;
            break;
        case  VDMOS_MOD_TRD2:
            model->VDMOStrd2 = value->rValue;
            model->VDMOStrd2Given = TRUE;
            break;
        case  VDMOS_MOD_TRG1:
            model->VDMOStrg1 = value->rValue;
            model->VDMOStrg1Given = TRUE;
            break;
        case  VDMOS_MOD_TRG2:
            model->VDMOStrg2 = value->rValue;
            model->VDMOStrg2Given = TRUE;
            break;
        case  VDMOS_MOD_TRS1:
            model->VDMOStrs1 = value->rValue;
            model->VDMOStrs1Given = TRUE;
            break;
        case  VDMOS_MOD_TRS2:
            model->VDMOStrs2 = value->rValue;
            model->VDMOStrs2Given = TRUE;
            break;
        case  VDIO_MOD_TRB1:
            model->VDIOtrb1 = value->rValue;
            model->VDIOtrb1Given = TRUE;
            break;
        case  VDIO_MOD_TRB2:
            model->VDIOtrb2 = value->rValue;
            model->VDIOtrb2Given = TRUE;
            break;
        case  VDMOS_MOD_TKSUBTHRES1:
            model->VDMOStksubthres1 = value->rValue;
            model->VDMOStksubthres1Given = TRUE;
            break;
        case  VDMOS_MOD_TKSUBTHRES2:
            model->VDMOStksubthres2 = value->rValue;
            model->VDMOStksubthres2Given = TRUE;
            break;
        case VDMOS_MOD_VGS_MAX:
            model->VDMOSvgsMax = value->rValue;
            model->VDMOSvgsMaxGiven = TRUE;
            break;
        case VDMOS_MOD_VGD_MAX:
            model->VDMOSvgdMax = value->rValue;
            model->VDMOSvgdMaxGiven = TRUE;
            break;
        case VDMOS_MOD_VDS_MAX:
            model->VDMOSvdsMax = value->rValue;
            model->VDMOSvdsMaxGiven = TRUE;
            break;
        case VDMOS_MOD_VGSR_MAX:
            model->VDMOSvgsrMax = value->rValue;
            model->VDMOSvgsrMaxGiven = TRUE;
            break;
        case VDMOS_MOD_VGDR_MAX:
            model->VDMOSvgdrMax = value->rValue;
            model->VDMOSvgdrMaxGiven = TRUE;
            break;
        case VDMOS_MOD_PD_MAX:
            model->VDMOSpd_max = value->rValue;
            model->VDMOSpd_maxGiven = TRUE;
            break;
        case VDMOS_MOD_ID_MAX:
            model->VDMOSid_max = value->rValue;
            model->VDMOSid_maxGiven = TRUE;
            break;
        case VDMOS_MOD_IDR_MAX:
            model->VDMOSidr_max = value->rValue;
            model->VDMOSidr_maxGiven = TRUE;
            break;
        case VDMOS_MOD_TE_MAX:
            model->VDMOSte_max = value->rValue;
            model->VDMOSte_maxGiven = TRUE;
            break;
        case VDMOS_MOD_RTH_EXT:
            model->VDMOSrth_ext = value->rValue;
            model->VDMOSrth_extGiven = TRUE;
            break;
        case VDMOS_MOD_DERATING:
            model->VDMOSderating = value->rValue;
            model->VDMOSderatingGiven = TRUE;
            break;
        default:
            return(E_BADPARM);
    }
    return(OK);
}
