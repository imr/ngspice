/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include "ngspice/const.h"
#include "ngspice/ifsim.h"
#include "mos1defs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int
MOS1mParam(int param, IFvalue *value, GENmodel *inModel)
{
    MOS1model *model = (MOS1model *)inModel;
    switch(param) {
        case MOS1_MOD_TNOM:
            model->MOS1tnom = value->rValue + CONSTCtoK;
            model->MOS1tnomGiven = TRUE;
            break;
        case MOS1_MOD_VTO:
            model->MOS1vt0 = value->rValue;
            model->MOS1vt0Given = TRUE;
            break;
        case MOS1_MOD_KP:
            model->MOS1transconductance = value->rValue;
            model->MOS1transconductanceGiven = TRUE;
            break;
        case MOS1_MOD_GAMMA:
            model->MOS1gamma = value->rValue;
            model->MOS1gammaGiven = TRUE;
            break;
        case MOS1_MOD_PHI:
            model->MOS1phi = value->rValue;
            model->MOS1phiGiven = TRUE;
            break;
        case MOS1_MOD_LAMBDA:
            model->MOS1lambda = value->rValue;
            model->MOS1lambdaGiven = TRUE;
            break;
        case MOS1_MOD_RD:
            model->MOS1drainResistance = value->rValue;
            model->MOS1drainResistanceGiven = TRUE;
            break;
        case MOS1_MOD_RS:
            model->MOS1sourceResistance = value->rValue;
            model->MOS1sourceResistanceGiven = TRUE;
            break;
        case MOS1_MOD_CBD:
            model->MOS1capBD = value->rValue;
            model->MOS1capBDGiven = TRUE;
            break;
        case MOS1_MOD_CBS:
            model->MOS1capBS = value->rValue;
            model->MOS1capBSGiven = TRUE;
            break;
        case MOS1_MOD_IS:
            model->MOS1jctSatCur = value->rValue;
            model->MOS1jctSatCurGiven = TRUE;
            break;
        case MOS1_MOD_PB:
            model->MOS1bulkJctPotential = value->rValue;
            model->MOS1bulkJctPotentialGiven = TRUE;
            break;
        case MOS1_MOD_CGSO:
            model->MOS1gateSourceOverlapCapFactor = value->rValue;
            model->MOS1gateSourceOverlapCapFactorGiven = TRUE;
            break;
        case MOS1_MOD_CGDO:
            model->MOS1gateDrainOverlapCapFactor = value->rValue;
            model->MOS1gateDrainOverlapCapFactorGiven = TRUE;
            break;
        case MOS1_MOD_CGBO:
            model->MOS1gateBulkOverlapCapFactor = value->rValue;
            model->MOS1gateBulkOverlapCapFactorGiven = TRUE;
            break;
        case MOS1_MOD_CJ:
            model->MOS1bulkCapFactor = value->rValue;
            model->MOS1bulkCapFactorGiven = TRUE;
            break;
        case MOS1_MOD_MJ:
            model->MOS1bulkJctBotGradingCoeff = value->rValue;
            model->MOS1bulkJctBotGradingCoeffGiven = TRUE;
            break;
        case MOS1_MOD_CJSW:
            model->MOS1sideWallCapFactor = value->rValue;
            model->MOS1sideWallCapFactorGiven = TRUE;
            break;
        case MOS1_MOD_MJSW:
            model->MOS1bulkJctSideGradingCoeff = value->rValue;
            model->MOS1bulkJctSideGradingCoeffGiven = TRUE;
            break;
        case MOS1_MOD_JS:
            model->MOS1jctSatCurDensity = value->rValue;
            model->MOS1jctSatCurDensityGiven = TRUE;
            break;
        case MOS1_MOD_TOX:
            model->MOS1oxideThickness = value->rValue;
            model->MOS1oxideThicknessGiven = TRUE;
            break;
        case MOS1_MOD_LD:
            model->MOS1latDiff = value->rValue;
            model->MOS1latDiffGiven = TRUE;
            break;
        case MOS1_MOD_RSH:
            model->MOS1sheetResistance = value->rValue;
            model->MOS1sheetResistanceGiven = TRUE;
            break;
        case MOS1_MOD_U0:
            model->MOS1surfaceMobility = value->rValue;
            model->MOS1surfaceMobilityGiven = TRUE;
            break;
        case MOS1_MOD_FC:
            model->MOS1fwdCapDepCoeff = value->rValue;
            model->MOS1fwdCapDepCoeffGiven = TRUE;
            break;
        case MOS1_MOD_NSS:
            model->MOS1surfaceStateDensity = value->rValue;
            model->MOS1surfaceStateDensityGiven = TRUE;
            break;
        case MOS1_MOD_NSUB:
            model->MOS1substrateDoping = value->rValue;
            model->MOS1substrateDopingGiven = TRUE;
            break;
        case MOS1_MOD_TPG:
            model->MOS1gateType = value->iValue;
            model->MOS1gateTypeGiven = TRUE;
            break;
        case MOS1_MOD_NMOS:
            if(value->iValue) {
                model->MOS1type = 1;
                model->MOS1typeGiven = TRUE;
            }
            break;
        case MOS1_MOD_PMOS:
            if(value->iValue) {
                model->MOS1type = -1;
                model->MOS1typeGiven = TRUE;
            }
            break;
	case MOS1_MOD_KF:
	    model->MOS1fNcoef = value->rValue;
	    model->MOS1fNcoefGiven = TRUE;
	    break;
	case MOS1_MOD_AF:
	    model->MOS1fNexp = value->rValue;
	    model->MOS1fNexpGiven = TRUE;
	    break;
        default:
            return(E_BADPARM);
    }
    return(OK);
}
