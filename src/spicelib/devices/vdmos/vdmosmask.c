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
        case VDMOS_MOD_TNOM:
            value->rValue = model->VDMOStnom-CONSTCtoK;
            return(OK);
        case VDMOS_MOD_AF:
            value->rValue = model->VDMOSfNcoef;
            return(OK);
        case VDMOS_MOD_KF:
            value->rValue = model->VDMOSfNexp;
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
        case VDIO_MOD_RB:
            value->rValue = model->VDIOresistance;
            return(OK);
        case VDIO_MOD_IS:
            value->rValue = model->VDIOjctSatCur;
            return(OK);
        case VDIO_MOD_N:
            value->rValue = model->VDIOn;
            return(OK);
        case VDIO_MOD_VJ:
            value->rValue = model->VDIOjunctionPot;
            return(OK);
        case VDIO_MOD_CJ:
            value->rValue = model->VDIOjunctionCap;
            return(OK);
        case VDIO_MOD_MJ:
            value->rValue = model->VDIOgradCoeff;
            return(OK);
        case VDIO_MOD_FC:
            value->rValue = model->VDIOdepletionCapCoeff;
            return(OK);
        case VDIO_MOD_BV:
            value->rValue = model->VDIObv;
            return(OK);
        case VDIO_MOD_IBV:
            value->rValue = model->VDIOibv;
            return(OK);
        case VDIO_MOD_NBV:
            value->rValue = model->VDIObrkdEmissionCoeff;
            return(OK);
        case VDMOS_MOD_RDS:
            value->rValue = model->VDMOSrds;
            return(OK);
        case VDIO_MOD_TT:
            value->rValue = model->VDIOtransitTime;
            return(OK);
        case VDIO_MOD_EG:
            value->rValue = model->VDIOeg;
            return(OK);
        case VDIO_MOD_XTI:
            value->rValue = model->VDIOxti;
            return(OK);
        case VDMOS_MOD_TCVTH:
            value->rValue = model->VDMOStcvth;
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
        case VDMOS_MOD_TRD1:
            value->rValue = model->VDMOStrd1; 
            return(OK);
        case VDMOS_MOD_TRD2:
            value->rValue = model->VDMOStrd2; 
            return(OK);
        case VDMOS_MOD_TRG1:
            value->rValue = model->VDMOStrg1; 
            return(OK);
        case VDMOS_MOD_TRG2:
            value->rValue = model->VDMOStrg2; 
            return(OK);
        case VDMOS_MOD_TRS1:
            value->rValue = model->VDMOStrs1; 
            return(OK);
        case VDMOS_MOD_TRS2:
            value->rValue = model->VDMOStrs2; 
            return(OK);
        case VDIO_MOD_TRB1:
            value->rValue = model->VDIOtrb1; 
            return(OK);
        case VDIO_MOD_TRB2:
            value->rValue = model->VDIOtrb2; 
            return(OK);
        case VDMOS_MOD_TKSUBTHRES1:
            value->rValue = model->VDMOStksubthres1; 
            return(OK);
        case VDMOS_MOD_TKSUBTHRES2:
            value->rValue = model->VDMOStksubthres2; 
            return(OK);
        /* SOA */
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
        case VDMOS_MOD_PD_MAX:
            value->rValue = model->VDMOSpd_max;
            return(OK);
        case VDMOS_MOD_ID_MAX:
            value->rValue = model->VDMOSid_max;
            return(OK);
        case VDMOS_MOD_IDR_MAX:
            value->rValue = model->VDMOSidr_max;
            return(OK);
        case VDMOS_MOD_TE_MAX:
            value->rValue = model->VDMOSte_max;
            return(OK);
        case VDMOS_MOD_RTH_EXT:
            value->rValue = model->VDMOSrth_ext;
            return(OK);
        case VDMOS_MOD_DERATING:
            value->rValue = model->VDMOSderating;
            return(OK);

        default:
            return(E_BADPARM);
    }
    /* NOTREACHED */
}

