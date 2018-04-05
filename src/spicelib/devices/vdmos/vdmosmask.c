/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1987 Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include "ngspice/const.h"
#include "ngspice/ifsim.h"
#include "ngspice/cktdefs.h"
#include "ngspice/devdefs.h"
#include "vdmosdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

/*ARGSUSED*/
int
VDMOSmAsk(CKTcircuit *ckt, GENmodel *inst, int which, IFvalue *value)
{
    VDMOSmodel *model = (VDMOSmodel *)inst;

    NG_IGNORE(ckt);

    switch(which) {
        case VDMOS_MOD_TNOM:
            value->rValue = model->VDMOStnom-CONSTCtoK;
            return(OK);
        case VDMOS_MOD_VTO:
            value->rValue = model->VDMOSvt0;
            return(OK);
        case VDMOS_MOD_KP:
            value->rValue = model->VDMOStransconductance;
            return(OK);
        case VDMOS_MOD_PHI:
            value->rValue = model->VDMOSphi;
            return(OK);
        case VDMOS_MOD_LAMBDA:
            value->rValue = model->VDMOSlambda;
            return(OK);
        case VDMOS_MOD_RD:
            value->rValue = model->VDMOSdrainResistance;
            return(OK);
        case VDMOS_MOD_RS:
            value->rValue = model->VDMOSsourceResistance;
            return(OK);
        case VDMOS_MOD_RG:
            value->rValue = model->VDMOSgateResistance;
            return(OK);
        case VDMOS_MOD_TYPE:
            if (model->VDMOStype > 0)
                value->sValue = "vdmosn";
            else
                value->sValue = "vdmosp";
            return(OK);
        case VDMOS_MOD_CGDMIN:
            value->rValue = model->VDMOScgdmin;
            return(OK);
        case VDMOS_MOD_CGDMAX:
            value->rValue = model->VDMOScgdmax;
            return(OK);
        case VDMOS_MOD_A:
            value->rValue = model->VDMOSa;
            return(OK);
        case VDMOS_MOD_CGS:
            value->rValue = model->VDMOScgs;
            return(OK);
        case VDMOS_MOD_IS:
            value->rValue = model->VDMOSjctSatCur;
            return(OK);
        case VDMOS_MOD_N:
            value->rValue = model->VDMOSDn;
            return(OK);
        case VDMOS_MOD_VJ:
            value->rValue = model->VDMOSbulkJctPotential;
            return(OK);
        case VDMOS_MOD_MJ:
            value->rValue = model->VDMOSbulkJctBotGradingCoeff;
            return(OK);
        case VDMOS_MOD_FC:
            value->rValue = model->VDMOSfwdCapDepCoeff;
            return(OK);
        case VDMOS_MOD_TT:
            value->rValue = model->VDIOtransitTime;
            return(OK);
        case VDMOS_MOD_EG:
            value->rValue = model->VDMOSDeg;
            return(OK);
        case VDMOS_MOD_XTI:
            value->rValue = model->VDMOSDxti;
            return(OK);
        default:
            return(E_BADPARM);
    }
    /* NOTREACHED */
}

