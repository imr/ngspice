/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include "ngspice/const.h"
#include "ngspice/ifsim.h"
#include "mos2defs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int
MOS2mAsk(CKTcircuit *ckt, GENmodel *inModel, int param,
         IFvalue *value)
{
    MOS2model *model = (MOS2model *)inModel;

    NG_IGNORE(ckt);

    switch(param) {
        case MOS2_MOD_TNOM:
            value->rValue = model->MOS2tnom - CONSTCtoK;
            break;
        case MOS2_MOD_VTO:
            value->rValue = model->MOS2vt0;
            break;
        case MOS2_MOD_KP:
            value->rValue = model->MOS2transconductance;
            break;
        case MOS2_MOD_GAMMA:
            value->rValue = model->MOS2gamma;
            break;
        case MOS2_MOD_PHI:
            value->rValue = model->MOS2phi;
            break;
        case MOS2_MOD_LAMBDA:
            value->rValue = model->MOS2lambda;
            break;
        case MOS2_MOD_RD:
            value->rValue = model->MOS2drainResistance;
            break;
        case MOS2_MOD_RS:
            value->rValue = model->MOS2sourceResistance;
            break;
        case MOS2_MOD_CBD:
            value->rValue = model->MOS2capBD;
            break;
        case MOS2_MOD_CBS:
            value->rValue = model->MOS2capBS;
            break;
        case MOS2_MOD_IS:
            value->rValue = model->MOS2jctSatCur;
            break;
        case MOS2_MOD_PB:
            value->rValue = model->MOS2bulkJctPotential;
            break;
        case MOS2_MOD_CGSO:
            value->rValue = model->MOS2gateSourceOverlapCapFactor;
            break;
        case MOS2_MOD_CGDO:
            value->rValue = model->MOS2gateDrainOverlapCapFactor;
            break;
        case MOS2_MOD_CGBO:
            value->rValue = model->MOS2gateBulkOverlapCapFactor;
            break;
        case MOS2_MOD_CJ:
            value->rValue = model->MOS2bulkCapFactor;
            break;
        case MOS2_MOD_MJ:
            value->rValue = model->MOS2bulkJctBotGradingCoeff;
            break;
        case MOS2_MOD_CJSW:
            value->rValue = model->MOS2sideWallCapFactor;
            break;
        case MOS2_MOD_MJSW:
            value->rValue = model->MOS2bulkJctSideGradingCoeff;
            break;
        case MOS2_MOD_JS:
            value->rValue = model->MOS2jctSatCurDensity;
            break;
        case MOS2_MOD_TOX:
            value->rValue = model->MOS2oxideThickness;
            break;
        case MOS2_MOD_LD:
            value->rValue = model->MOS2latDiff;
            break;
        case MOS2_MOD_RSH:
            value->rValue = model->MOS2sheetResistance;
            break;
        case MOS2_MOD_U0:
            value->rValue = model->MOS2surfaceMobility;
            break;
        case MOS2_MOD_FC:
            value->rValue = model->MOS2fwdCapDepCoeff;
            break;
        case MOS2_MOD_NSUB:
            value->rValue = model->MOS2substrateDoping;
            break;
        case MOS2_MOD_TPG:
            value->rValue = model->MOS2gateType;
            break;
        case MOS2_MOD_NSS:
            value->rValue = model->MOS2surfaceStateDensity;
            break;
        case MOS2_MOD_NFS:
            value->rValue = model->MOS2fastSurfaceStateDensity;
            break;
        case MOS2_MOD_DELTA:
            value->rValue = model->MOS2narrowFactor;
            break;
        case MOS2_MOD_UEXP:
            value->rValue = model->MOS2critFieldExp;
            break;
        case MOS2_MOD_VMAX:
            value->rValue = model->MOS2maxDriftVel;
            break;
        case MOS2_MOD_XJ:
            value->rValue = model->MOS2junctionDepth;
            break;
        case MOS2_MOD_NEFF:
            value->rValue = model->MOS2channelCharge;
            break;
        case MOS2_MOD_UCRIT:
            value->rValue = model->MOS2critField;
            break;
	case MOS2_MOD_KF:
	    value->rValue = model->MOS2fNcoef;
	    break;
	case MOS2_MOD_AF:
	    value->rValue = model->MOS2fNexp;
	    break;
	case MOS2_MOD_TYPE:
	    if (model->MOS2type > 0)
	        value->sValue = "nmos";
	    else
	        value->sValue = "pmos";
	    break;
        default:
            return(E_BADPARM);
    }
    return(OK);
}
