/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1987 Thomas L. Quarles
Modified: 2000 AlansFixes
**********/
/*
 */

#include "ngspice/ngspice.h"
#include "ngspice/const.h"
#include "ngspice/ifsim.h"
#include "ngspice/cktdefs.h"
#include "ngspice/devdefs.h"
#include "mos3defs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


/*ARGSUSED*/
int
MOS3mAsk(CKTcircuit *ckt, GENmodel *inModel, int which, IFvalue *value)
{
    MOS3model *model = (MOS3model *)inModel;

    NG_IGNORE(ckt);

    switch(which) {
        case MOS3_MOD_TNOM:
            value->rValue = model->MOS3tnom-CONSTCtoK;
            return(OK);
        case MOS3_MOD_VTO:
            value->rValue = model->MOS3vt0;
            return(OK);
        case MOS3_MOD_KP:
            value->rValue = model->MOS3transconductance;
            return(OK);
        case MOS3_MOD_GAMMA:
            value->rValue = model->MOS3gamma;
            return(OK);
        case MOS3_MOD_PHI:
            value->rValue = model->MOS3phi;
            return(OK);
        case MOS3_MOD_RD:
            value->rValue = model->MOS3drainResistance;
            return(OK);
        case MOS3_MOD_RS:
            value->rValue = model->MOS3sourceResistance;
            return(OK);
        case MOS3_MOD_CBD:
            value->rValue = model->MOS3capBD;
            return(OK);
        case MOS3_MOD_CBS:
            value->rValue = model->MOS3capBS;
            return(OK);
        case MOS3_MOD_IS:
            value->rValue = model->MOS3jctSatCur;
            return(OK);
        case MOS3_MOD_PB:
            value->rValue = model->MOS3bulkJctPotential;
            return(OK);
        case MOS3_MOD_CGSO:
            value->rValue = model->MOS3gateSourceOverlapCapFactor;
            return(OK);
        case MOS3_MOD_CGDO:
            value->rValue = model->MOS3gateDrainOverlapCapFactor;
            return(OK);
        case MOS3_MOD_CGBO:
            value->rValue = model->MOS3gateBulkOverlapCapFactor;
            return(OK);
        case MOS3_MOD_CJ:
            value->rValue = model->MOS3bulkCapFactor;
            return(OK);
        case MOS3_MOD_MJ:
            value->rValue = model->MOS3bulkJctBotGradingCoeff;
            return(OK);
        case MOS3_MOD_CJSW:
            value->rValue = model->MOS3sideWallCapFactor;
            return(OK);
        case MOS3_MOD_MJSW:
            value->rValue = model->MOS3bulkJctSideGradingCoeff;
            return(OK);
        case MOS3_MOD_JS:
            value->rValue = model->MOS3jctSatCurDensity;
            return(OK);
        case MOS3_MOD_TOX:
            value->rValue = model->MOS3oxideThickness;
            return(OK);
        case MOS3_MOD_LD:
            value->rValue = model->MOS3latDiff;
            return(OK);
        case MOS3_MOD_XL:
            value->rValue = model->MOS3lengthAdjust;
            return(OK);
        case MOS3_MOD_WD:
            value->rValue = model->MOS3widthNarrow;
            return(OK);
        case MOS3_MOD_XW:
            value->rValue = model->MOS3widthAdjust;
            return(OK);
        case MOS3_MOD_DELVTO:
            value->rValue = model->MOS3delvt0;
            return(OK);
        case MOS3_MOD_RSH:
            value->rValue = model->MOS3sheetResistance;
            return(OK);
        case MOS3_MOD_U0:
            value->rValue = model->MOS3surfaceMobility;
            return(OK);
        case MOS3_MOD_FC:
            value->rValue = model->MOS3fwdCapDepCoeff;
            return(OK);
        case MOS3_MOD_NSUB:
            value->rValue = model->MOS3substrateDoping;
            return(OK);
        case MOS3_MOD_TPG:
            value->iValue = model->MOS3gateType;
            return(OK);
        case MOS3_MOD_NSS:
            value->rValue = model->MOS3surfaceStateDensity;
            return(OK);
        case MOS3_MOD_NFS:
            value->rValue = model->MOS3fastSurfaceStateDensity;
            return(OK);
        case MOS3_MOD_DELTA:
            value->rValue = model->MOS3narrowFactor;
            return(OK);
        case MOS3_MOD_VMAX:
            value->rValue = model->MOS3maxDriftVel;
            return(OK);
        case MOS3_MOD_XJ:
            value->rValue = model->MOS3junctionDepth;
            return(OK);
        case MOS3_MOD_ETA:
            value->rValue = model->MOS3eta;
            return(OK);
        case MOS3_MOD_XD:
            value->rValue = model->MOS3coeffDepLayWidth;
            return(OK);
        case MOS3_DELTA:
            value->rValue = model->MOS3delta;
            return(OK);
        case MOS3_MOD_THETA:
            value->rValue = model->MOS3theta;
            return(OK);
        case MOS3_MOD_ALPHA:
            value->rValue = model->MOS3alpha;
            return(OK);
        case MOS3_MOD_KAPPA:
            value->rValue = model->MOS3kappa;
            return(OK);
        case MOS3_MOD_KF:
            value->rValue = model->MOS3fNcoef;
            return(OK);
        case MOS3_MOD_AF:
            value->rValue = model->MOS3fNexp;
            return(OK);
        case MOS3_MOD_NLEV:
            value->iValue = model->MOS3nlev;
            return(OK);
        case MOS3_MOD_GDSNOI:
            value->rValue = model->MOS3gdsnoi;
            return(OK);
        case MOS3_MOD_TYPE:
            if (model->MOS3type > 0)
                value->sValue = "nmos";
            else
                value->sValue = "pmos";
            return(OK);
        default:
            return(E_BADPARM);
    }
    /* NOTREACHED */
}

