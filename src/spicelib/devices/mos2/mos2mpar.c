/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

#include "ngspice/ngspice.h"
#include "ngspice/const.h"
#include "ngspice/ifsim.h"
#include "mos2defs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
MOS2mParam(int param, IFvalue *value, GENmodel *inModel)
{
    MOS2model *model = (MOS2model *)inModel;
    switch(param) {
        case MOS2_MOD_TNOM:
            model->MOS2tnom = value->rValue+CONSTCtoK;
            model->MOS2tnomGiven = TRUE;
            break;
        case MOS2_MOD_VTO:
            model->MOS2vt0 = value->rValue;
            model->MOS2vt0Given = TRUE;
            break;
        case MOS2_MOD_KP:
            model->MOS2transconductance = value->rValue;
            model->MOS2transconductanceGiven = TRUE;
            break;
        case MOS2_MOD_GAMMA:
            model->MOS2gamma = value->rValue;
            model->MOS2gammaGiven = TRUE;
            break;
        case MOS2_MOD_PHI:
            model->MOS2phi = value->rValue;
            model->MOS2phiGiven = TRUE;
            break;
        case MOS2_MOD_LAMBDA:
            model->MOS2lambda = value->rValue;
            model->MOS2lambdaGiven = TRUE;
            break;
        case MOS2_MOD_RD:
            model->MOS2drainResistance = value->rValue;
            model->MOS2drainResistanceGiven = TRUE;
            break;
        case MOS2_MOD_RS:
            model->MOS2sourceResistance = value->rValue;
            model->MOS2sourceResistanceGiven = TRUE;
            break;
        case MOS2_MOD_CBD:
            model->MOS2capBD = value->rValue;
            model->MOS2capBDGiven = TRUE;
            break;
        case MOS2_MOD_CBS:
            model->MOS2capBS = value->rValue;
            model->MOS2capBSGiven = TRUE;
            break;
        case MOS2_MOD_IS:
            model->MOS2jctSatCur = value->rValue;
            model->MOS2jctSatCurGiven = TRUE;
            break;
        case MOS2_MOD_PB:
            model->MOS2bulkJctPotential = value->rValue;
            model->MOS2bulkJctPotentialGiven = TRUE;
            break;
        case MOS2_MOD_CGSO:
            model->MOS2gateSourceOverlapCapFactor = value->rValue;
            model->MOS2gateSourceOverlapCapFactorGiven = TRUE;
            break;
        case MOS2_MOD_CGDO:
            model->MOS2gateDrainOverlapCapFactor = value->rValue;
            model->MOS2gateDrainOverlapCapFactorGiven = TRUE;
            break;
        case MOS2_MOD_CGBO:
            model->MOS2gateBulkOverlapCapFactor = value->rValue;
            model->MOS2gateBulkOverlapCapFactorGiven = TRUE;
            break;
        case MOS2_MOD_CJ:
            model->MOS2bulkCapFactor = value->rValue;
            model->MOS2bulkCapFactorGiven = TRUE;
            break;
        case MOS2_MOD_MJ:
            model->MOS2bulkJctBotGradingCoeff = value->rValue;
            model->MOS2bulkJctBotGradingCoeffGiven = TRUE;
            break;
        case MOS2_MOD_CJSW:
            model->MOS2sideWallCapFactor = value->rValue;
            model->MOS2sideWallCapFactorGiven = TRUE;
            break;
        case MOS2_MOD_MJSW:
            model->MOS2bulkJctSideGradingCoeff = value->rValue;
            model->MOS2bulkJctSideGradingCoeffGiven = TRUE;
            break;
        case MOS2_MOD_JS:
            model->MOS2jctSatCurDensity = value->rValue;
            model->MOS2jctSatCurDensityGiven = TRUE;
            break;
        case MOS2_MOD_TOX:
            model->MOS2oxideThickness = value->rValue;
            model->MOS2oxideThicknessGiven = TRUE;
            break;
        case MOS2_MOD_LD:
            model->MOS2latDiff = value->rValue;
            model->MOS2latDiffGiven = TRUE;
            break;
        case MOS2_MOD_RSH:
            model->MOS2sheetResistance = value->rValue;
            model->MOS2sheetResistanceGiven = TRUE;
            break;
        case MOS2_MOD_U0:
            model->MOS2surfaceMobility = value->rValue;
            model->MOS2surfaceMobilityGiven = TRUE;
            break;
        case MOS2_MOD_FC:
            model->MOS2fwdCapDepCoeff = value->rValue;
            model->MOS2fwdCapDepCoeffGiven = TRUE;
            break;
        case MOS2_MOD_NSUB:
            model->MOS2substrateDoping = value->rValue;
            model->MOS2substrateDopingGiven = TRUE;
            break;
        case MOS2_MOD_TPG:
            model->MOS2gateType = value->iValue;
            model->MOS2gateTypeGiven = TRUE;
            break;
        case MOS2_MOD_NSS:
            model->MOS2surfaceStateDensity = value->rValue;
            model->MOS2surfaceStateDensityGiven = TRUE;
            break;
        case MOS2_MOD_NFS:
            model->MOS2fastSurfaceStateDensity = value->rValue;
            model->MOS2fastSurfaceStateDensityGiven = TRUE;
            break;
        case MOS2_MOD_DELTA:
            model->MOS2narrowFactor = value->rValue;
            model->MOS2narrowFactorGiven = TRUE;
            break;
        case MOS2_MOD_UEXP:
            model->MOS2critFieldExp = value->rValue;
            model->MOS2critFieldExpGiven = TRUE;
            break;
        case MOS2_MOD_VMAX:
            model->MOS2maxDriftVel = value->rValue;
            model->MOS2maxDriftVelGiven = TRUE;
            break;
        case MOS2_MOD_XJ:
            model->MOS2junctionDepth = value->rValue;
            model->MOS2junctionDepthGiven = TRUE;
            break;
        case MOS2_MOD_NEFF:
            model->MOS2channelCharge = value->rValue;
            model->MOS2channelChargeGiven = TRUE;
            break;
        case MOS2_MOD_UCRIT:
            model->MOS2critField = value->rValue;
            model->MOS2critFieldGiven = TRUE;
            break;
        case MOS2_MOD_NMOS:
            if(value->iValue) {
                model->MOS2type = 1;
                model->MOS2typeGiven = TRUE;
            }
            break;
        case MOS2_MOD_PMOS:
            if(value->iValue) {
                model->MOS2type = -1;
                model->MOS2typeGiven = TRUE;
            }
            break;
	case MOS2_MOD_KF:
	    model->MOS2fNcoef = value->rValue;
	    model->MOS2fNcoefGiven = TRUE;
	    break;
	case MOS2_MOD_AF:
	    model->MOS2fNexp = value->rValue;
	    model->MOS2fNexpGiven = TRUE;
	    break;
        default:
            return(E_BADPARM);
    }
    return(OK);
}
