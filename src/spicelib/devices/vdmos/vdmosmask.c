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
        case VDMOS_MOD_VTH:
            value->rValue = model->VDMOSvth0;
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
        case VDMOS_MOD_THETA:
            value->rValue = model->VDMOStheta;
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
        case VDMOS_MOD_RQ:
            value->rValue = model->VDMOSqsResistance;
            return(OK);
        case VDMOS_MOD_VQ:
            value->rValue = model->VDMOSqsVoltage;
            return(OK);
        case VDMOS_MOD_MTRIODE:
            value->rValue = model->VDMOSmtr;
            return(OK);
        case VDMOS_MOD_SUBSHIFT:
            value->rValue = model->VDMOSsubshift;
            return(OK);
        case VDMOS_MOD_KSUBTHRES:
            value->rValue = model->VDMOSksubthres;
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

        /* body diode */
        case VDMOS_MOD_RB:
            value->rValue = model->VDIOresistance;
            return(OK);
        case VDMOS_MOD_IS:
            value->rValue = model->VDIOjctSatCur;
            return(OK);
        case VDMOS_MOD_N:
            value->rValue = model->VDMOSn;
            return(OK);
        case VDMOS_MOD_VJ:
            value->rValue = model->VDIOjunctionPot;
            return(OK);
        case VDMOS_MOD_CJ:
            value->rValue = model->VDIOjunctionCap;
            return(OK);
        case VDMOS_MOD_MJ:
            value->rValue = model->VDIOgradCoeff;
            return(OK);
        case VDMOS_MOD_BV:
            value->rValue = model->VDMOSbv;
            return(OK);
        case VDMOS_MOD_IBV:
            value->rValue = model->VDMOSibv;
            return(OK);
        case VDMOS_MOD_NBV:
            value->rValue = model->VDIObrkdEmissionCoeff;
            return(OK);
        case VDMOS_MOD_RDS:
            value->rValue = model->VDMOSrds;
            return(OK);
        case VDMOS_MOD_FC:
            value->rValue = model->VDIOdepletionCapCoeff;
            return(OK);
        case VDMOS_MOD_TT:
            value->rValue = model->VDIOtransitTime;
            return(OK);
        case VDMOS_MOD_EG:
            value->rValue = model->VDMOSeg;
            return(OK);
        case VDMOS_MOD_XTI:
            value->rValue = model->VDMOSxti;
            return(OK);
        case VDMOS_MOD_RTHJC:
            value->rValue = model->VDMOSrthjc; 
            return(OK);
        case VDMOS_MOD_RTHCA:
            value->rValue = model->VDMOSrthca; 
            return(OK);
        case VDMOS_MOD_CTHJ:
            value->rValue = model->VDMOScthj; 
            return(OK);
        case VDMOS_MOD_MU:
            value->rValue = model->VDMOSmu; 
            return(OK);
        case VDMOS_MOD_TEXP0:
            value->rValue = model->VDMOStexp0; 
            return(OK);
        case VDMOS_MOD_TEXP1:
            value->rValue = model->VDMOStexp1; 
            return(OK);
        case VDMOS_MOD_TCVTH:
            value->rValue = model->VDMOStcvth;
            return(OK);
        case VDMOS_MOD_VGS_MAX:
            value->rValue = model->VDMOSvgsMax;
            return(OK);
        case VDMOS_MOD_VGD_MAX:
            value->rValue = model->VDMOSvgdMax;
            return(OK);
        case VDMOS_MOD_VDS_MAX:
            value->rValue = model->VDMOSvdsMax;
            return(OK);
        case VDMOS_MOD_VGSR_MAX:
            value->rValue = model->VDMOSvgsrMax;
            return(OK);
        case VDMOS_MOD_VGDR_MAX:
            value->rValue = model->VDMOSvgdrMax;
            return(OK);
        default:
            return(E_BADPARM);
    }
    /* NOTREACHED */
}

