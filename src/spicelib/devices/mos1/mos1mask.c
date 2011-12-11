/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1987 Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include "ngspice/const.h"
#include "ngspice/ifsim.h"
#include "ngspice/cktdefs.h"
#include "ngspice/devdefs.h"
#include "mos1defs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

/*ARGSUSED*/
int
MOS1mAsk(CKTcircuit *ckt, GENmodel *inst, int which, IFvalue *value)
{
    MOS1model *model = (MOS1model *)inst;

    NG_IGNORE(ckt);

    switch(which) {
        case MOS1_MOD_TNOM:
            value->rValue = model->MOS1tnom-CONSTCtoK;
            return(OK);
        case MOS1_MOD_VTO:
            value->rValue = model->MOS1vt0;
            return(OK);
        case MOS1_MOD_KP:
            value->rValue = model->MOS1transconductance;
            return(OK);
        case MOS1_MOD_GAMMA:
            value->rValue = model->MOS1gamma;
            return(OK);
        case MOS1_MOD_PHI:
            value->rValue = model->MOS1phi;
            return(OK);
        case MOS1_MOD_LAMBDA:
            value->rValue = model->MOS1lambda;
            return(OK);
        case MOS1_MOD_RD:
            value->rValue = model->MOS1drainResistance;
            return(OK);
        case MOS1_MOD_RS:
            value->rValue = model->MOS1sourceResistance;
            return(OK);
        case MOS1_MOD_CBD:
            value->rValue = model->MOS1capBD;
            return(OK);
        case MOS1_MOD_CBS:
            value->rValue = model->MOS1capBS;
            return(OK);
        case MOS1_MOD_IS:
            value->rValue = model->MOS1jctSatCur;
            return(OK);
        case MOS1_MOD_PB:
            value->rValue = model->MOS1bulkJctPotential;
            return(OK);
        case MOS1_MOD_CGSO:
            value->rValue = model->MOS1gateSourceOverlapCapFactor;
            return(OK);
        case MOS1_MOD_CGDO:
            value->rValue = model->MOS1gateDrainOverlapCapFactor;
            return(OK);
        case MOS1_MOD_CGBO:
            value->rValue = model->MOS1gateBulkOverlapCapFactor;
            return(OK);
        case MOS1_MOD_CJ:
            value->rValue = model->MOS1bulkCapFactor;
            return(OK);
        case MOS1_MOD_MJ:
            value->rValue = model->MOS1bulkJctBotGradingCoeff;
            return(OK);
        case MOS1_MOD_CJSW:
            value->rValue = model->MOS1sideWallCapFactor;
            return(OK);
        case MOS1_MOD_MJSW:
            value->rValue = model->MOS1bulkJctSideGradingCoeff;
            return(OK);
        case MOS1_MOD_JS:
            value->rValue = model->MOS1jctSatCurDensity;
            return(OK);
        case MOS1_MOD_TOX:
            value->rValue = model->MOS1oxideThickness;
            return(OK);
        case MOS1_MOD_LD:
            value->rValue = model->MOS1latDiff;
            return(OK);
        case MOS1_MOD_RSH:
            value->rValue = model->MOS1sheetResistance;
            return(OK);
        case MOS1_MOD_U0:
            value->rValue = model->MOS1surfaceMobility;
            return(OK);
        case MOS1_MOD_FC:
            value->rValue = model->MOS1fwdCapDepCoeff;
            return(OK);
        case MOS1_MOD_NSUB:
            value->rValue = model->MOS1substrateDoping;
            return(OK);
        case MOS1_MOD_TPG:
            value->iValue = model->MOS1gateType;
            return(OK);
        case MOS1_MOD_NSS:
            value->rValue = model->MOS1surfaceStateDensity;
            return(OK);
        case MOS1_MOD_TYPE:
	    if (model->MOS1type > 0)
		value->sValue = "nmos";
	    else
		value->sValue = "pmos";
            return(OK);
        default:
            return(E_BADPARM);
    }
    /* NOTREACHED */
}

