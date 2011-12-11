/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1987 Thomas L. Quarles
Modified: Alan Gillespie
**********/
/*
 */

#include "ngspice/ngspice.h"
#include "ngspice/const.h"
#include "ngspice/ifsim.h"
#include "ngspice/cktdefs.h"
#include "ngspice/devdefs.h"
#include "mos9defs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


/*ARGSUSED*/
int
MOS9mAsk(CKTcircuit *ckt, GENmodel *inst, int which, IFvalue *value)
{
    MOS9model *here = (MOS9model *)inst;

    NG_IGNORE(ckt);

    switch(which) {
        case MOS9_MOD_TNOM:
            value->rValue = here->MOS9tnom-CONSTCtoK;
            return(OK);
        case MOS9_MOD_VTO:
            value->rValue = here->MOS9vt0;
            return(OK);
        case MOS9_MOD_KP:
            value->rValue = here->MOS9transconductance;
            return(OK);
        case MOS9_MOD_GAMMA:
            value->rValue = here->MOS9gamma;
            return(OK);
        case MOS9_MOD_PHI:
            value->rValue = here->MOS9phi;
            return(OK);
        case MOS9_MOD_RD:
            value->rValue = here->MOS9drainResistance;
            return(OK);
        case MOS9_MOD_RS:
            value->rValue = here->MOS9sourceResistance;
            return(OK);
        case MOS9_MOD_CBD:
            value->rValue = here->MOS9capBD;
            return(OK);
        case MOS9_MOD_CBS:
            value->rValue = here->MOS9capBS;
            return(OK);
        case MOS9_MOD_IS:
            value->rValue = here->MOS9jctSatCur;
            return(OK);
        case MOS9_MOD_PB:
            value->rValue = here->MOS9bulkJctPotential;
            return(OK);
        case MOS9_MOD_CGSO:
            value->rValue = here->MOS9gateSourceOverlapCapFactor;
            return(OK);
        case MOS9_MOD_CGDO:
            value->rValue = here->MOS9gateDrainOverlapCapFactor;
            return(OK);
        case MOS9_MOD_CGBO:
            value->rValue = here->MOS9gateBulkOverlapCapFactor;
            return(OK);
        case MOS9_MOD_CJ:
            value->rValue = here->MOS9bulkCapFactor;
            return(OK);
        case MOS9_MOD_MJ:
            value->rValue = here->MOS9bulkJctBotGradingCoeff;
            return(OK);
        case MOS9_MOD_CJSW:
            value->rValue = here->MOS9sideWallCapFactor;
            return(OK);
        case MOS9_MOD_MJSW:
            value->rValue = here->MOS9bulkJctSideGradingCoeff;
            return(OK);
        case MOS9_MOD_JS:
            value->rValue = here->MOS9jctSatCurDensity;
            return(OK);
        case MOS9_MOD_TOX:
            value->rValue = here->MOS9oxideThickness;
            return(OK);
        case MOS9_MOD_LD:
            value->rValue = here->MOS9latDiff;
            return(OK);
        case MOS9_MOD_XL:
            value->rValue = here->MOS9lengthAdjust;
            return(OK);
        case MOS9_MOD_WD:
            value->rValue = here->MOS9widthNarrow;
            return(OK);
        case MOS9_MOD_XW:
            value->rValue = here->MOS9widthAdjust;
            return(OK);
        case MOS9_MOD_DELVTO:
            value->rValue = here->MOS9delvt0;
            return(OK);
        case MOS9_MOD_RSH:
            value->rValue = here->MOS9sheetResistance;
            return(OK);
        case MOS9_MOD_U0:
            value->rValue = here->MOS9surfaceMobility;
            return(OK);
        case MOS9_MOD_FC:
            value->rValue = here->MOS9fwdCapDepCoeff;
            return(OK);
        case MOS9_MOD_NSUB:
            value->rValue = here->MOS9substrateDoping;
            return(OK);
        case MOS9_MOD_TPG:
            value->iValue = here->MOS9gateType;
            return(OK);
        case MOS9_MOD_NSS:
            value->rValue = here->MOS9surfaceStateDensity;
            return(OK);
        case MOS9_MOD_NFS:
            value->rValue = here->MOS9fastSurfaceStateDensity;
            return(OK);
        case MOS9_MOD_DELTA:
            value->rValue = here->MOS9narrowFactor;
            return(OK);
        case MOS9_MOD_VMAX:
            value->rValue = here->MOS9maxDriftVel;
            return(OK);
        case MOS9_MOD_XJ:
            value->rValue = here->MOS9junctionDepth;
            return(OK);
        case MOS9_MOD_ETA:
            value->rValue = here->MOS9eta;
            return(OK);
        case MOS9_MOD_XD:
            value->rValue = here->MOS9coeffDepLayWidth;
            return(OK);
        case MOS9_DELTA:
            value->rValue = here->MOS9delta;
            return(OK);
        case MOS9_MOD_THETA:
            value->rValue = here->MOS9theta;
            return(OK);
        case MOS9_MOD_ALPHA:
            value->rValue = here->MOS9alpha;
            return(OK);
        case MOS9_MOD_KAPPA:
            value->rValue = here->MOS9kappa;
            return(OK);
        case MOS9_MOD_KF:
            value->rValue = here->MOS9fNcoef;
            return(OK);
        case MOS9_MOD_AF:
            value->rValue = here->MOS9fNexp;
            return(OK);

	case MOS9_MOD_TYPE:
	    if (here->MOS9type > 0)
	        value->sValue = "nmos";
	    else
	        value->sValue = "pmos";
            return(OK);
        default:
            return(E_BADPARM);
    }
    /* NOTREACHED */
}

