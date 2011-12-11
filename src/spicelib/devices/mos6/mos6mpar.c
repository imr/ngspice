/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1989 Takayasu Sakurai
**********/
/*
 */

#include "ngspice/ngspice.h"
#include "ngspice/const.h"
#include "ngspice/ifsim.h"
#include "mos6defs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
MOS6mParam(int param, IFvalue *value, GENmodel *inModel)
{
    MOS6model *model = (MOS6model *)inModel;
    switch(param) {
        case MOS6_MOD_TNOM:
            model->MOS6tnom = value->rValue+CONSTCtoK;
            model->MOS6tnomGiven = TRUE;
            break;
        case MOS6_MOD_VTO:
            model->MOS6vt0 = value->rValue;
            model->MOS6vt0Given = TRUE;
            break;
        case MOS6_MOD_KV:
            model->MOS6kv = value->rValue;
            model->MOS6kvGiven = TRUE;
            break;
        case MOS6_MOD_NV:
            model->MOS6nv = value->rValue;
            model->MOS6nvGiven = TRUE;
            break;
        case MOS6_MOD_KC:
            model->MOS6kc = value->rValue;
            model->MOS6kcGiven = TRUE;
            break;
        case MOS6_MOD_NC:
            model->MOS6nc = value->rValue;
            model->MOS6ncGiven = TRUE;
            break;
        case MOS6_MOD_NVTH:
            model->MOS6nvth = value->rValue;
            model->MOS6nvthGiven = TRUE;
            break;
        case MOS6_MOD_PS:
            model->MOS6ps = value->rValue;
            model->MOS6psGiven = TRUE;
            break;
        case MOS6_MOD_GAMMA:
            model->MOS6gamma = value->rValue;
            model->MOS6gammaGiven = TRUE;
            break;
        case MOS6_MOD_GAMMA1:
            model->MOS6gamma1 = value->rValue;
            model->MOS6gamma1Given = TRUE;
            break;
        case MOS6_MOD_SIGMA:
            model->MOS6sigma = value->rValue;
            model->MOS6sigmaGiven = TRUE;
            break;
        case MOS6_MOD_PHI:
            model->MOS6phi = value->rValue;
            model->MOS6phiGiven = TRUE;
            break;
        case MOS6_MOD_LAMBDA:
            model->MOS6lambda = value->rValue;
            model->MOS6lambdaGiven = TRUE;
            break;
        case MOS6_MOD_LAMDA0:
            model->MOS6lamda0 = value->rValue;
            model->MOS6lamda0Given = TRUE;
            break;
        case MOS6_MOD_LAMDA1:
            model->MOS6lamda1 = value->rValue;
            model->MOS6lamda1Given = TRUE;
            break;
        case MOS6_MOD_RD:
            model->MOS6drainResistance = value->rValue;
            model->MOS6drainResistanceGiven = TRUE;
            break;
        case MOS6_MOD_RS:
            model->MOS6sourceResistance = value->rValue;
            model->MOS6sourceResistanceGiven = TRUE;
            break;
        case MOS6_MOD_CBD:
            model->MOS6capBD = value->rValue;
            model->MOS6capBDGiven = TRUE;
            break;
        case MOS6_MOD_CBS:
            model->MOS6capBS = value->rValue;
            model->MOS6capBSGiven = TRUE;
            break;
        case MOS6_MOD_IS:
            model->MOS6jctSatCur = value->rValue;
            model->MOS6jctSatCurGiven = TRUE;
            break;
        case MOS6_MOD_PB:
            model->MOS6bulkJctPotential = value->rValue;
            model->MOS6bulkJctPotentialGiven = TRUE;
            break;
        case MOS6_MOD_CGSO:
            model->MOS6gateSourceOverlapCapFactor = value->rValue;
            model->MOS6gateSourceOverlapCapFactorGiven = TRUE;
            break;
        case MOS6_MOD_CGDO:
            model->MOS6gateDrainOverlapCapFactor = value->rValue;
            model->MOS6gateDrainOverlapCapFactorGiven = TRUE;
            break;
        case MOS6_MOD_CGBO:
            model->MOS6gateBulkOverlapCapFactor = value->rValue;
            model->MOS6gateBulkOverlapCapFactorGiven = TRUE;
            break;
        case MOS6_MOD_CJ:
            model->MOS6bulkCapFactor = value->rValue;
            model->MOS6bulkCapFactorGiven = TRUE;
            break;
        case MOS6_MOD_MJ:
            model->MOS6bulkJctBotGradingCoeff = value->rValue;
            model->MOS6bulkJctBotGradingCoeffGiven = TRUE;
            break;
        case MOS6_MOD_CJSW:
            model->MOS6sideWallCapFactor = value->rValue;
            model->MOS6sideWallCapFactorGiven = TRUE;
            break;
        case MOS6_MOD_MJSW:
            model->MOS6bulkJctSideGradingCoeff = value->rValue;
            model->MOS6bulkJctSideGradingCoeffGiven = TRUE;
            break;
        case MOS6_MOD_JS:
            model->MOS6jctSatCurDensity = value->rValue;
            model->MOS6jctSatCurDensityGiven = TRUE;
            break;
        case MOS6_MOD_TOX:
            model->MOS6oxideThickness = value->rValue;
            model->MOS6oxideThicknessGiven = TRUE;
            break;
        case MOS6_MOD_LD:
            model->MOS6latDiff = value->rValue;
            model->MOS6latDiffGiven = TRUE;
            break;
        case MOS6_MOD_RSH:
            model->MOS6sheetResistance = value->rValue;
            model->MOS6sheetResistanceGiven = TRUE;
            break;
        case MOS6_MOD_U0:
            model->MOS6surfaceMobility = value->rValue;
            model->MOS6surfaceMobilityGiven = TRUE;
            break;
        case MOS6_MOD_FC:
            model->MOS6fwdCapDepCoeff = value->rValue;
            model->MOS6fwdCapDepCoeffGiven = TRUE;
            break;
        case MOS6_MOD_NSS:
            model->MOS6surfaceStateDensity = value->rValue;
            model->MOS6surfaceStateDensityGiven = TRUE;
            break;
        case MOS6_MOD_NSUB:
            model->MOS6substrateDoping = value->rValue;
            model->MOS6substrateDopingGiven = TRUE;
            break;
        case MOS6_MOD_TPG:
            model->MOS6gateType = value->iValue;
            model->MOS6gateTypeGiven = TRUE;
            break;
        case MOS6_MOD_NMOS:
            if(value->iValue) {
                model->MOS6type = 1;
                model->MOS6typeGiven = TRUE;
            }
            break;
        case MOS6_MOD_PMOS:
            if(value->iValue) {
                model->MOS6type = -1;
                model->MOS6typeGiven = TRUE;
            }
            break;
        default:
            return(E_BADPARM);
    }
    return(OK);
}
