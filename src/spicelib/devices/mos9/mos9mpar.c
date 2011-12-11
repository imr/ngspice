/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: Alan Gillespie
**********/
/*
 */

#include "ngspice/ngspice.h"
#include "ngspice/const.h"
#include "ngspice/ifsim.h"
#include "mos9defs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
MOS9mParam(int param, IFvalue *value, GENmodel *inModel)
{
    register MOS9model *model = (MOS9model *)inModel;
    switch(param) {
        case MOS9_MOD_VTO:
            model->MOS9vt0 = value->rValue;
            model->MOS9vt0Given = TRUE;
            break;
        case MOS9_MOD_KP:
            model->MOS9transconductance = value->rValue;
            model->MOS9transconductanceGiven = TRUE;
            break;
        case MOS9_MOD_GAMMA:
            model->MOS9gamma = value->rValue;
            model->MOS9gammaGiven = TRUE;
            break;
        case MOS9_MOD_PHI:
            model->MOS9phi = value->rValue;
            model->MOS9phiGiven = TRUE;
            break;
        case MOS9_MOD_RD:
            model->MOS9drainResistance = value->rValue;
            model->MOS9drainResistanceGiven = TRUE;
            break;
        case MOS9_MOD_RS:
            model->MOS9sourceResistance = value->rValue;
            model->MOS9sourceResistanceGiven = TRUE;
            break;
        case MOS9_MOD_CBD:
            model->MOS9capBD = value->rValue;
            model->MOS9capBDGiven = TRUE;
            break;
        case MOS9_MOD_CBS:
            model->MOS9capBS = value->rValue;
            model->MOS9capBSGiven = TRUE;
            break;
        case MOS9_MOD_IS:
            model->MOS9jctSatCur = value->rValue;
            model->MOS9jctSatCurGiven = TRUE;
            break;
        case MOS9_MOD_PB:
            model->MOS9bulkJctPotential = value->rValue;
            model->MOS9bulkJctPotentialGiven = TRUE;
            break;
        case MOS9_MOD_CGSO:
            model->MOS9gateSourceOverlapCapFactor = value->rValue;
            model->MOS9gateSourceOverlapCapFactorGiven = TRUE;
            break;
        case MOS9_MOD_CGDO:
            model->MOS9gateDrainOverlapCapFactor = value->rValue;
            model->MOS9gateDrainOverlapCapFactorGiven = TRUE;
            break;
        case MOS9_MOD_CGBO:
            model->MOS9gateBulkOverlapCapFactor = value->rValue;
            model->MOS9gateBulkOverlapCapFactorGiven = TRUE;
            break;
        case MOS9_MOD_RSH:
            model->MOS9sheetResistance = value->rValue;
            model->MOS9sheetResistanceGiven = TRUE;
            break;
        case MOS9_MOD_CJ:
            model->MOS9bulkCapFactor = value->rValue;
            model->MOS9bulkCapFactorGiven = TRUE;
            break;
        case MOS9_MOD_MJ:
            model->MOS9bulkJctBotGradingCoeff = value->rValue;
            model->MOS9bulkJctBotGradingCoeffGiven = TRUE;
            break;
        case MOS9_MOD_CJSW:
            model->MOS9sideWallCapFactor = value->rValue;
            model->MOS9sideWallCapFactorGiven = TRUE;
            break;
        case MOS9_MOD_MJSW:
            model->MOS9bulkJctSideGradingCoeff = value->rValue;
            model->MOS9bulkJctSideGradingCoeffGiven = TRUE;
            break;
        case MOS9_MOD_JS:
            model->MOS9jctSatCurDensity = value->rValue;
            model->MOS9jctSatCurDensityGiven = TRUE;
            break;
        case MOS9_MOD_TOX:
            model->MOS9oxideThickness = value->rValue;
            model->MOS9oxideThicknessGiven = TRUE;
            break;
        case MOS9_MOD_LD:
            model->MOS9latDiff = value->rValue;
            model->MOS9latDiffGiven = TRUE;
            break;
        case MOS9_MOD_XL:
            model->MOS9lengthAdjust = value->rValue;
            model->MOS9lengthAdjustGiven = TRUE;
            break;
        case MOS9_MOD_WD:
            model->MOS9widthNarrow = value->rValue;
            model->MOS9widthNarrowGiven = TRUE;
            break;
        case MOS9_MOD_XW:
            model->MOS9widthAdjust = value->rValue;
            model->MOS9widthAdjustGiven = TRUE;
            break;
        case MOS9_MOD_DELVTO:
            model->MOS9delvt0 = value->rValue;
            model->MOS9delvt0Given = TRUE;
            break;
        case MOS9_MOD_U0:
            model->MOS9surfaceMobility = value->rValue;
            model->MOS9surfaceMobilityGiven = TRUE;
            break;
        case MOS9_MOD_FC:
            model->MOS9fwdCapDepCoeff = value->rValue;
            model->MOS9fwdCapDepCoeffGiven = TRUE;
            break;
        case MOS9_MOD_NSUB:
            model->MOS9substrateDoping = value->rValue;
            model->MOS9substrateDopingGiven = TRUE;
            break;
        case MOS9_MOD_TPG:
            model->MOS9gateType = value->iValue;
            model->MOS9gateTypeGiven = TRUE;
            break;
        case MOS9_MOD_NSS:
            model->MOS9surfaceStateDensity = value->rValue;
            model->MOS9surfaceStateDensityGiven = TRUE;
            break;
        case MOS9_MOD_ETA:
            model->MOS9eta = value->rValue;
            model->MOS9etaGiven = TRUE;
            break;
        case MOS9_MOD_DELTA:
            model->MOS9delta = value->rValue;
            model->MOS9deltaGiven = TRUE;
            break;
        case MOS9_MOD_NFS:
            model->MOS9fastSurfaceStateDensity = value->rValue;
            model->MOS9fastSurfaceStateDensityGiven = TRUE;
            break;
        case MOS9_MOD_THETA:
            model->MOS9theta = value->rValue;
            model->MOS9thetaGiven = TRUE;
            break;
        case MOS9_MOD_VMAX:
            model->MOS9maxDriftVel = value->rValue;
            model->MOS9maxDriftVelGiven = TRUE;
            break;
        case MOS9_MOD_KAPPA:
            model->MOS9kappa = value->rValue;
            model->MOS9kappaGiven = TRUE;
            break;
        case MOS9_MOD_NMOS:
            if(value->iValue) {
                model->MOS9type = 1;
                model->MOS9typeGiven = TRUE;
            }
            break;
        case MOS9_MOD_PMOS:
            if(value->iValue) {
                model->MOS9type = -1;
                model->MOS9typeGiven = TRUE;
            }
            break;
        case MOS9_MOD_XJ:
            model->MOS9junctionDepth = value->rValue;
            model->MOS9junctionDepthGiven = TRUE;
            break;
        case MOS9_MOD_TNOM:
            model->MOS9tnom = value->rValue+CONSTCtoK;
            model->MOS9tnomGiven = TRUE;
            break;
	case MOS9_MOD_KF:
	    model->MOS9fNcoef = value->rValue;
	    model->MOS9fNcoefGiven = TRUE;
	    break;
	case MOS9_MOD_AF:
	    model->MOS9fNexp = value->rValue;
	    model->MOS9fNexpGiven = TRUE;
	    break;
        default:
            return(E_BADPARM);
    }
    return(OK);
}
