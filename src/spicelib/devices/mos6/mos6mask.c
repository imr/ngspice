/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1989 Takayasu Sakurai
**********/

#include "ngspice/ngspice.h"
#include "ngspice/const.h"
#include "ngspice/ifsim.h"
#include "mos6defs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int
MOS6mAsk(CKTcircuit *ckt, GENmodel *inModel, int param,
         IFvalue *value)
{
    MOS6model *model = (MOS6model *)inModel;

    NG_IGNORE(ckt);

    switch(param) {
        case MOS6_MOD_TNOM:
            value->rValue = model->MOS6tnom;
            break;
        case MOS6_MOD_VTO:
            value->rValue = model->MOS6vt0;
            break;
        case MOS6_MOD_KV:
            value->rValue = model->MOS6kv;
            break;
        case MOS6_MOD_NV:
            value->rValue = model->MOS6nv;
            break;
        case MOS6_MOD_KC:
            value->rValue = model->MOS6kc;
            break;
        case MOS6_MOD_NC:
            value->rValue = model->MOS6nc;
            break;
        case MOS6_MOD_NVTH:
            value->rValue = model->MOS6nvth;
            break;
        case MOS6_MOD_PS:
            value->rValue = model->MOS6ps;
            break;
        case MOS6_MOD_GAMMA:
            value->rValue = model->MOS6gamma;
            break;
        case MOS6_MOD_GAMMA1:
            value->rValue = model->MOS6gamma1;
            break;
        case MOS6_MOD_SIGMA:
            value->rValue = model->MOS6sigma;
            break;
        case MOS6_MOD_PHI:
            value->rValue = model->MOS6phi;
            break;
        case MOS6_MOD_LAMBDA:
            value->rValue = model->MOS6lambda;
            break;
        case MOS6_MOD_LAMDA0:
            value->rValue = model->MOS6lamda0;
            break;
        case MOS6_MOD_LAMDA1:
            value->rValue = model->MOS6lamda1;
            break;
        case MOS6_MOD_RD:
            value->rValue = model->MOS6drainResistance;
            break;
        case MOS6_MOD_RS:
            value->rValue = model->MOS6sourceResistance;
            break;
        case MOS6_MOD_CBD:
            value->rValue = model->MOS6capBD;
            break;
        case MOS6_MOD_CBS:
            value->rValue = model->MOS6capBS;
            break;
        case MOS6_MOD_IS:
            value->rValue = model->MOS6jctSatCur;
            break;
        case MOS6_MOD_PB:
            value->rValue = model->MOS6bulkJctPotential;
            break;
        case MOS6_MOD_CGSO:
            value->rValue = model->MOS6gateSourceOverlapCapFactor;
            break;
        case MOS6_MOD_CGDO:
            value->rValue = model->MOS6gateDrainOverlapCapFactor;
            break;
        case MOS6_MOD_CGBO:
            value->rValue = model->MOS6gateBulkOverlapCapFactor;
            break;
        case MOS6_MOD_CJ:
            value->rValue = model->MOS6bulkCapFactor;
            break;
        case MOS6_MOD_MJ:
            value->rValue = model->MOS6bulkJctBotGradingCoeff;
            break;
        case MOS6_MOD_CJSW:
            value->rValue = model->MOS6sideWallCapFactor;
            break;
        case MOS6_MOD_MJSW:
            value->rValue = model->MOS6bulkJctSideGradingCoeff;
            break;
        case MOS6_MOD_JS:
            value->rValue = model->MOS6jctSatCurDensity;
            break;
        case MOS6_MOD_TOX:
            value->rValue = model->MOS6oxideThickness;
            break;
        case MOS6_MOD_LD:
            value->rValue = model->MOS6latDiff;
            break;
        case MOS6_MOD_RSH:
            value->rValue = model->MOS6sheetResistance;
            break;
        case MOS6_MOD_U0:
            value->rValue = model->MOS6surfaceMobility;
            break;
        case MOS6_MOD_FC:
            value->rValue = model->MOS6fwdCapDepCoeff;
            break;
        case MOS6_MOD_NSS:
            value->rValue = model->MOS6surfaceStateDensity;
            break;
        case MOS6_MOD_NSUB:
            value->rValue = model->MOS6substrateDoping;
            break;
        case MOS6_MOD_TPG:
            value->iValue = model->MOS6gateType;
            break;
	case MOS6_MOD_TYPE:
	    if (model->MOS6type > 0)
	        value->sValue = "nmos";
	    else
	        value->sValue = "pmos";
	    break;
        default:
            return(E_BADPARM);
    }
    return(OK);
}
