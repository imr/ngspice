/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: 2000 AlansFixes
**********/
/*
 */

#include "ngspice/ngspice.h"
#include "ngspice/const.h"
#include "ngspice/ifsim.h"
#include "mos3defs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
MOS3mParam(int param, IFvalue *value, GENmodel *inModel)
{
    MOS3model *model = (MOS3model *)inModel;
    switch(param) {
        case MOS3_MOD_VTO:
            model->MOS3vt0 = value->rValue;
            model->MOS3vt0Given = TRUE;
            break;
        case MOS3_MOD_KP:
            model->MOS3transconductance = value->rValue;
            model->MOS3transconductanceGiven = TRUE;
            break;
        case MOS3_MOD_GAMMA:
            model->MOS3gamma = value->rValue;
            model->MOS3gammaGiven = TRUE;
            break;
        case MOS3_MOD_PHI:
            model->MOS3phi = value->rValue;
            model->MOS3phiGiven = TRUE;
            break;
        case MOS3_MOD_RD:
            model->MOS3drainResistance = value->rValue;
            model->MOS3drainResistanceGiven = TRUE;
            break;
        case MOS3_MOD_RS:
            model->MOS3sourceResistance = value->rValue;
            model->MOS3sourceResistanceGiven = TRUE;
            break;
        case MOS3_MOD_CBD:
            model->MOS3capBD = value->rValue;
            model->MOS3capBDGiven = TRUE;
            break;
        case MOS3_MOD_CBS:
            model->MOS3capBS = value->rValue;
            model->MOS3capBSGiven = TRUE;
            break;
        case MOS3_MOD_IS:
            model->MOS3jctSatCur = value->rValue;
            model->MOS3jctSatCurGiven = TRUE;
            break;
        case MOS3_MOD_PB:
            model->MOS3bulkJctPotential = value->rValue;
            model->MOS3bulkJctPotentialGiven = TRUE;
            break;
        case MOS3_MOD_CGSO:
            model->MOS3gateSourceOverlapCapFactor = value->rValue;
            model->MOS3gateSourceOverlapCapFactorGiven = TRUE;
            break;
        case MOS3_MOD_CGDO:
            model->MOS3gateDrainOverlapCapFactor = value->rValue;
            model->MOS3gateDrainOverlapCapFactorGiven = TRUE;
            break;
        case MOS3_MOD_CGBO:
            model->MOS3gateBulkOverlapCapFactor = value->rValue;
            model->MOS3gateBulkOverlapCapFactorGiven = TRUE;
            break;
        case MOS3_MOD_RSH:
            model->MOS3sheetResistance = value->rValue;
            model->MOS3sheetResistanceGiven = TRUE;
            break;
        case MOS3_MOD_CJ:
            model->MOS3bulkCapFactor = value->rValue;
            model->MOS3bulkCapFactorGiven = TRUE;
            break;
        case MOS3_MOD_MJ:
            model->MOS3bulkJctBotGradingCoeff = value->rValue;
            model->MOS3bulkJctBotGradingCoeffGiven = TRUE;
            break;
        case MOS3_MOD_CJSW:
            model->MOS3sideWallCapFactor = value->rValue;
            model->MOS3sideWallCapFactorGiven = TRUE;
            break;
        case MOS3_MOD_MJSW:
            model->MOS3bulkJctSideGradingCoeff = value->rValue;
            model->MOS3bulkJctSideGradingCoeffGiven = TRUE;
            break;
        case MOS3_MOD_JS:
            model->MOS3jctSatCurDensity = value->rValue;
            model->MOS3jctSatCurDensityGiven = TRUE;
            break;
        case MOS3_MOD_TOX:
            model->MOS3oxideThickness = value->rValue;
            model->MOS3oxideThicknessGiven = TRUE;
            break;
        case MOS3_MOD_LD:
            model->MOS3latDiff = value->rValue;
            model->MOS3latDiffGiven = TRUE;
            break;
        case MOS3_MOD_XL:
            model->MOS3lengthAdjust = value->rValue;
            model->MOS3lengthAdjustGiven = TRUE;
            break;
        case MOS3_MOD_WD:
            model->MOS3widthNarrow = value->rValue;
            model->MOS3widthNarrowGiven = TRUE;
            break;
        case MOS3_MOD_XW:
            model->MOS3widthAdjust = value->rValue;
            model->MOS3widthAdjustGiven = TRUE;
            break;
        case MOS3_MOD_DELVTO:
            model->MOS3delvt0 = value->rValue;
            model->MOS3delvt0Given = TRUE;
            break;    
        case MOS3_MOD_U0:
            model->MOS3surfaceMobility = value->rValue;
            model->MOS3surfaceMobilityGiven = TRUE;
            break;
        case MOS3_MOD_FC:
            model->MOS3fwdCapDepCoeff = value->rValue;
            model->MOS3fwdCapDepCoeffGiven = TRUE;
            break;
        case MOS3_MOD_NSUB:
            model->MOS3substrateDoping = value->rValue;
            model->MOS3substrateDopingGiven = TRUE;
            break;
        case MOS3_MOD_TPG:
            model->MOS3gateType = value->iValue;
            model->MOS3gateTypeGiven = TRUE;
            break;
        case MOS3_MOD_NSS:
            model->MOS3surfaceStateDensity = value->rValue;
            model->MOS3surfaceStateDensityGiven = TRUE;
            break;
        case MOS3_MOD_ETA:
            model->MOS3eta = value->rValue;
            model->MOS3etaGiven = TRUE;
            break;
        case MOS3_MOD_DELTA:
            model->MOS3delta = value->rValue;
            model->MOS3deltaGiven = TRUE;
            break;
        case MOS3_MOD_NFS:
            model->MOS3fastSurfaceStateDensity = value->rValue;
            model->MOS3fastSurfaceStateDensityGiven = TRUE;
            break;
        case MOS3_MOD_THETA:
            model->MOS3theta = value->rValue;
            model->MOS3thetaGiven = TRUE;
            break;
        case MOS3_MOD_VMAX:
            model->MOS3maxDriftVel = value->rValue;
            model->MOS3maxDriftVelGiven = TRUE;
            break;
        case MOS3_MOD_KAPPA:
            model->MOS3kappa = value->rValue;
            model->MOS3kappaGiven = TRUE;
            break;
        case MOS3_MOD_NMOS:
            if(value->iValue) {
                model->MOS3type = 1;
                model->MOS3typeGiven = TRUE;
            }
            break;
        case MOS3_MOD_PMOS:
            if(value->iValue) {
                model->MOS3type = -1;
                model->MOS3typeGiven = TRUE;
            }
            break;
        case MOS3_MOD_XJ:
            model->MOS3junctionDepth = value->rValue;
            model->MOS3junctionDepthGiven = TRUE;
            break;
        case MOS3_MOD_TNOM:
            model->MOS3tnom = value->rValue+CONSTCtoK;
            model->MOS3tnomGiven = TRUE;
            break;
	case MOS3_MOD_KF:
	    model->MOS3fNcoef = value->rValue;
	    model->MOS3fNcoefGiven = TRUE;
	    break;
	case MOS3_MOD_AF:
	    model->MOS3fNexp = value->rValue;
	    model->MOS3fNexpGiven = TRUE;
	    break;
        default:
            return(E_BADPARM);
    }
    return(OK);
}
