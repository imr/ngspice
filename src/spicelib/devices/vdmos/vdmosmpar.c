/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
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
        case VDMOS_MOD_VTO:
            model->VDMOSvt0 = value->rValue;
            model->VDMOSvt0Given = TRUE;
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
        case VDMOS_MOD_RB:
            model->VDIOresistance = value->rValue;
            model->VDIOresistanceGiven = TRUE;
            model->VDIOresistTemp1 = 0;
            model->VDIOresistTemp2 = 0;
            break;
        case VDMOS_MOD_IS:
            model->VDMOSjctSatCur = value->rValue;
            model->VDMOSjctSatCurGiven = TRUE;
            break;
        case VDMOS_MOD_VJ:
            model->VDMOSbulkJctPotential = value->rValue;
            model->VDMOSbulkJctPotentialGiven = TRUE;
            break;
        case VDMOS_MOD_MJ:
            model->VDMOSbulkJctBotGradingCoeff = value->rValue;
            model->VDMOSbulkJctBotGradingCoeffGiven = TRUE;
            break;
        case VDMOS_MOD_FC:
            model->VDMOSfwdCapDepCoeff = value->rValue;
            model->VDMOSfwdCapDepCoeffGiven = TRUE;
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
        case VDMOS_MOD_BV:
            model->VDMOSDbv = value->rValue;
            model->VDMOSDbvGiven = TRUE;
            break;
        case VDMOS_MOD_IBV:
            model->VDMOSDibv = value->rValue;
            model->VDMOSDibvGiven = TRUE;
            break;
        case VDMOS_MOD_NBV:
            model->VDIObrkdEmissionCoeff = value->rValue;
            model->VDIObrkdEmissionCoeffGiven = TRUE;
            break;
        case VDMOS_MOD_N:
            model->VDMOSDn = value->rValue;
            model->VDMOSDnGiven = TRUE;
            break;
        case VDMOS_MOD_TT:
            model->VDIOtransitTime = value->rValue;
            model->VDIOtransitTimeGiven = TRUE;
            break;
        case VDMOS_MOD_EG:
            model->VDMOSDeg = value->rValue;
            model->VDMOSDegGiven = TRUE;
            break;
        case VDMOS_MOD_XTI:
            model->VDMOSDxti = value->rValue;
            model->VDMOSDxtiGiven = TRUE;
            break;
        default:
            return(E_BADPARM);
    }
    return(OK);
}
