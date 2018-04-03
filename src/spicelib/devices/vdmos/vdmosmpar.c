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
        case VDMOS_MOD_GAMMA:
            model->VDMOSgamma = value->rValue;
            model->VDMOSgammaGiven = TRUE;
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
        case VDMOS_MOD_CBD:
            model->VDMOScapBD = value->rValue;
            model->VDMOScapBDGiven = TRUE;
            break;
        case VDMOS_MOD_CBS:
            model->VDMOScapBS = value->rValue;
            model->VDMOScapBSGiven = TRUE;
            break;
        case VDMOS_MOD_IS:
            model->VDMOSjctSatCur = value->rValue;
            model->VDMOSjctSatCurGiven = TRUE;
            break;
        case VDMOS_MOD_VJ:
            model->VDMOSbulkJctPotential = value->rValue;
            model->VDMOSbulkJctPotentialGiven = TRUE;
            break;
        case VDMOS_MOD_CJ:
            model->VDMOSbulkCapFactor = value->rValue;
            model->VDMOSbulkCapFactorGiven = TRUE;
            break;
        case VDMOS_MOD_MJ:
            model->VDMOSbulkJctBotGradingCoeff = value->rValue;
            model->VDMOSbulkJctBotGradingCoeffGiven = TRUE;
            break;
        case VDMOS_MOD_CJSW:
            model->VDMOSsideWallCapFactor = value->rValue;
            model->VDMOSsideWallCapFactorGiven = TRUE;
            break;
        case VDMOS_MOD_MJSW:
            model->VDMOSbulkJctSideGradingCoeff = value->rValue;
            model->VDMOSbulkJctSideGradingCoeffGiven = TRUE;
            break;
        case VDMOS_MOD_JS:
            model->VDMOSjctSatCurDensity = value->rValue;
            model->VDMOSjctSatCurDensityGiven = TRUE;
            break;
        case VDMOS_MOD_TOX:
            model->VDMOSoxideThickness = value->rValue;
            model->VDMOSoxideThicknessGiven = TRUE;
            break;
        case VDMOS_MOD_LD:
            model->VDMOSlatDiff = value->rValue;
            model->VDMOSlatDiffGiven = TRUE;
            break;
        case VDMOS_MOD_RSH:
            model->VDMOSsheetResistance = value->rValue;
            model->VDMOSsheetResistanceGiven = TRUE;
            break;
        case VDMOS_MOD_U0:
            model->VDMOSsurfaceMobility = value->rValue;
            model->VDMOSsurfaceMobilityGiven = TRUE;
            break;
        case VDMOS_MOD_FC:
            model->VDMOSfwdCapDepCoeff = value->rValue;
            model->VDMOSfwdCapDepCoeffGiven = TRUE;
            break;
        case VDMOS_MOD_NSS:
            model->VDMOSsurfaceStateDensity = value->rValue;
            model->VDMOSsurfaceStateDensityGiven = TRUE;
            break;
        case VDMOS_MOD_NSUB:
            model->VDMOSsubstrateDoping = value->rValue;
            model->VDMOSsubstrateDopingGiven = TRUE;
            break;
        case VDMOS_MOD_TPG:
            model->VDMOSgateType = value->iValue;
            model->VDMOSgateTypeGiven = TRUE;
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
        default:
            return(E_BADPARM);
    }
    return(OK);
}
