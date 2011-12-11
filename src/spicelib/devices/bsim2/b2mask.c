/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1988 Hong J. Park
**********/
/*
 */

#include "ngspice/ngspice.h"
#include "ngspice/ifsim.h"
#include "ngspice/cktdefs.h"
#include "ngspice/devdefs.h"
#include "bsim2def.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


/*ARGSUSED*/
int
B2mAsk(CKTcircuit *ckt, GENmodel *inst, int which, IFvalue *value)
{
    B2model *model = (B2model *)inst;

    NG_IGNORE(ckt);

        switch(which) {
        case BSIM2_MOD_VFB0: 
            value->rValue = model->B2vfb0; 
            return(OK);
        case  BSIM2_MOD_VFBL :
          value->rValue = model->B2vfbL;
            return(OK);
        case  BSIM2_MOD_VFBW :
          value->rValue = model->B2vfbW;
            return(OK);
        case  BSIM2_MOD_PHI0 :
          value->rValue = model->B2phi0;
            return(OK);
        case  BSIM2_MOD_PHIL :
          value->rValue = model->B2phiL;
            return(OK);
        case  BSIM2_MOD_PHIW :
          value->rValue = model->B2phiW;
            return(OK);
        case  BSIM2_MOD_K10 :
          value->rValue = model->B2k10;
            return(OK);
        case  BSIM2_MOD_K1L :
          value->rValue = model->B2k1L;
            return(OK);
        case  BSIM2_MOD_K1W :
          value->rValue = model->B2k1W;
            return(OK);
        case  BSIM2_MOD_K20 :
          value->rValue = model->B2k20;
            return(OK);
        case  BSIM2_MOD_K2L :
          value->rValue = model->B2k2L;
            return(OK);
        case  BSIM2_MOD_K2W :
          value->rValue = model->B2k2W;
            return(OK);
        case  BSIM2_MOD_ETA00 :
          value->rValue = model->B2eta00;
            return(OK);
        case  BSIM2_MOD_ETA0L :
          value->rValue = model->B2eta0L;
            return(OK);
        case  BSIM2_MOD_ETA0W :
          value->rValue = model->B2eta0W;
            return(OK);
        case  BSIM2_MOD_ETAB0 :
          value->rValue = model->B2etaB0;
            return(OK);
        case  BSIM2_MOD_ETABL :
          value->rValue = model->B2etaBL;
            return(OK);
        case  BSIM2_MOD_ETABW :
          value->rValue = model->B2etaBW;
            return(OK);
        case  BSIM2_MOD_DELTAL :
          value->rValue = model->B2deltaL =  value->rValue;
            return(OK);
        case  BSIM2_MOD_DELTAW :
          value->rValue = model->B2deltaW =  value->rValue;
            return(OK);
        case  BSIM2_MOD_MOB00 :
          value->rValue = model->B2mob00;
            return(OK);
        case  BSIM2_MOD_MOB0B0 :
          value->rValue = model->B2mob0B0;
            return(OK);
        case  BSIM2_MOD_MOB0BL :
          value->rValue = model->B2mob0BL;
            return(OK);
        case  BSIM2_MOD_MOB0BW :
          value->rValue = model->B2mob0BW;
            return(OK);
        case  BSIM2_MOD_MOBS00 :
          value->rValue = model->B2mobs00;
            return(OK);
        case  BSIM2_MOD_MOBS0L :
          value->rValue = model->B2mobs0L;
            return(OK);
        case  BSIM2_MOD_MOBS0W :
          value->rValue = model->B2mobs0W;
            return(OK);
        case  BSIM2_MOD_MOBSB0 :
          value->rValue = model->B2mobsB0;
            return(OK);
        case  BSIM2_MOD_MOBSBL :
          value->rValue = model->B2mobsBL;
            return(OK);
        case  BSIM2_MOD_MOBSBW :
          value->rValue = model->B2mobsBW;
            return(OK);
        case  BSIM2_MOD_MOB200 :
          value->rValue = model->B2mob200;
            return(OK);
        case  BSIM2_MOD_MOB20L :
          value->rValue = model->B2mob20L;
            return(OK);
        case  BSIM2_MOD_MOB20W :
          value->rValue = model->B2mob20W;
            return(OK);
        case  BSIM2_MOD_MOB2B0 :
          value->rValue = model->B2mob2B0;
            return(OK);
        case  BSIM2_MOD_MOB2BL :
          value->rValue = model->B2mob2BL;
            return(OK);
        case  BSIM2_MOD_MOB2BW :
          value->rValue = model->B2mob2BW;
            return(OK);
        case  BSIM2_MOD_MOB2G0 :
          value->rValue = model->B2mob2G0;
            return(OK);
        case  BSIM2_MOD_MOB2GL :
          value->rValue = model->B2mob2GL;
            return(OK);
        case  BSIM2_MOD_MOB2GW :
          value->rValue = model->B2mob2GW;
            return(OK);
        case  BSIM2_MOD_MOB300 :
          value->rValue = model->B2mob300;
            return(OK);
        case  BSIM2_MOD_MOB30L :
          value->rValue = model->B2mob30L;
            return(OK);
        case  BSIM2_MOD_MOB30W :
          value->rValue = model->B2mob30W;
            return(OK);
        case  BSIM2_MOD_MOB3B0 :
          value->rValue = model->B2mob3B0;
            return(OK);
        case  BSIM2_MOD_MOB3BL :
          value->rValue = model->B2mob3BL;
            return(OK);
        case  BSIM2_MOD_MOB3BW :
          value->rValue = model->B2mob3BW;
            return(OK);
        case  BSIM2_MOD_MOB3G0 :
          value->rValue = model->B2mob3G0;
            return(OK);
        case  BSIM2_MOD_MOB3GL :
          value->rValue = model->B2mob3GL;
            return(OK);
        case  BSIM2_MOD_MOB3GW :
          value->rValue = model->B2mob3GW;
            return(OK);
        case  BSIM2_MOD_MOB400 :
          value->rValue = model->B2mob400;
            return(OK);
        case  BSIM2_MOD_MOB40L :
          value->rValue = model->B2mob40L;
            return(OK);
        case  BSIM2_MOD_MOB40W :
          value->rValue = model->B2mob40W;
            return(OK);
        case  BSIM2_MOD_MOB4B0 :
          value->rValue = model->B2mob4B0;
            return(OK);
        case  BSIM2_MOD_MOB4BL :
          value->rValue = model->B2mob4BL;
            return(OK);
        case  BSIM2_MOD_MOB4BW :
          value->rValue = model->B2mob4BW;
            return(OK);
        case  BSIM2_MOD_MOB4G0 :
          value->rValue = model->B2mob4G0;
            return(OK);
        case  BSIM2_MOD_MOB4GL :
          value->rValue = model->B2mob4GL;
            return(OK);
        case  BSIM2_MOD_MOB4GW :
          value->rValue = model->B2mob4GW;
            return(OK);
        case  BSIM2_MOD_UA00 :
          value->rValue = model->B2ua00;
            return(OK);
        case  BSIM2_MOD_UA0L :
          value->rValue = model->B2ua0L;
            return(OK);
        case  BSIM2_MOD_UA0W :
          value->rValue = model->B2ua0W;
            return(OK);
        case  BSIM2_MOD_UAB0 :
          value->rValue = model->B2uaB0;
            return(OK);
        case  BSIM2_MOD_UABL :
          value->rValue = model->B2uaBL;
            return(OK);
        case  BSIM2_MOD_UABW :
          value->rValue = model->B2uaBW;
            return(OK);
        case  BSIM2_MOD_UB00 :
          value->rValue = model->B2ub00;
            return(OK);
        case  BSIM2_MOD_UB0L :
          value->rValue = model->B2ub0L;
            return(OK);
        case  BSIM2_MOD_UB0W :
          value->rValue = model->B2ub0W;
            return(OK);
        case  BSIM2_MOD_UBB0 :
          value->rValue = model->B2ubB0;
            return(OK);
        case  BSIM2_MOD_UBBL :
          value->rValue = model->B2ubBL;
            return(OK);
        case  BSIM2_MOD_UBBW :
          value->rValue = model->B2ubBW;
            return(OK);
        case  BSIM2_MOD_U100 :
          value->rValue = model->B2u100;
            return(OK);
        case  BSIM2_MOD_U10L :
          value->rValue = model->B2u10L;
            return(OK);
        case  BSIM2_MOD_U10W :
          value->rValue = model->B2u10W;
            return(OK);
        case  BSIM2_MOD_U1B0 :
          value->rValue = model->B2u1B0;
            return(OK);
        case  BSIM2_MOD_U1BL :
          value->rValue = model->B2u1BL;
            return(OK);
        case  BSIM2_MOD_U1BW :
          value->rValue = model->B2u1BW;
            return(OK);
        case  BSIM2_MOD_U1D0 :
          value->rValue = model->B2u1D0;
            return(OK);
        case  BSIM2_MOD_U1DL :
          value->rValue = model->B2u1DL;
            return(OK);
        case  BSIM2_MOD_U1DW :
          value->rValue = model->B2u1DW;
            return(OK);
        case  BSIM2_MOD_N00 :
          value->rValue = model->B2n00;
            return(OK);
        case  BSIM2_MOD_N0L :
          value->rValue = model->B2n0L;
            return(OK);
        case  BSIM2_MOD_N0W :
          value->rValue = model->B2n0W;
            return(OK);
        case  BSIM2_MOD_NB0 :
          value->rValue = model->B2nB0;
            return(OK);
        case  BSIM2_MOD_NBL :
          value->rValue = model->B2nBL;
            return(OK);
        case  BSIM2_MOD_NBW :
          value->rValue = model->B2nBW;
            return(OK);
        case  BSIM2_MOD_ND0 :
          value->rValue = model->B2nD0;
            return(OK);
        case  BSIM2_MOD_NDL :
          value->rValue = model->B2nDL;
            return(OK);
        case  BSIM2_MOD_NDW :
          value->rValue = model->B2nDW;
            return(OK);
        case  BSIM2_MOD_VOF00 :
          value->rValue = model->B2vof00;
            return(OK);
        case  BSIM2_MOD_VOF0L :
          value->rValue = model->B2vof0L;
            return(OK);
        case  BSIM2_MOD_VOF0W :
          value->rValue = model->B2vof0W;
            return(OK);
        case  BSIM2_MOD_VOFB0 :
          value->rValue = model->B2vofB0;
            return(OK);
        case  BSIM2_MOD_VOFBL :
          value->rValue = model->B2vofBL;
            return(OK);
        case  BSIM2_MOD_VOFBW :
          value->rValue = model->B2vofBW;
            return(OK);
        case  BSIM2_MOD_VOFD0 :
          value->rValue = model->B2vofD0;
            return(OK);
        case  BSIM2_MOD_VOFDL :
          value->rValue = model->B2vofDL;
            return(OK);
        case  BSIM2_MOD_VOFDW :
          value->rValue = model->B2vofDW;
            return(OK);
        case  BSIM2_MOD_AI00 :
          value->rValue = model->B2ai00;
            return(OK);
        case  BSIM2_MOD_AI0L :
          value->rValue = model->B2ai0L;
            return(OK);
        case  BSIM2_MOD_AI0W :
          value->rValue = model->B2ai0W;
            return(OK);
        case  BSIM2_MOD_AIB0 :
          value->rValue = model->B2aiB0;
            return(OK);
        case  BSIM2_MOD_AIBL :
          value->rValue = model->B2aiBL;
            return(OK);
        case  BSIM2_MOD_AIBW :
          value->rValue = model->B2aiBW;
            return(OK);
        case  BSIM2_MOD_BI00 :
          value->rValue = model->B2bi00;
            return(OK);
        case  BSIM2_MOD_BI0L :
          value->rValue = model->B2bi0L;
            return(OK);
        case  BSIM2_MOD_BI0W :
          value->rValue = model->B2bi0W;
            return(OK);
        case  BSIM2_MOD_BIB0 :
          value->rValue = model->B2biB0;
            return(OK);
        case  BSIM2_MOD_BIBL :
          value->rValue = model->B2biBL;
            return(OK);
        case  BSIM2_MOD_BIBW :
          value->rValue = model->B2biBW;
            return(OK);
        case  BSIM2_MOD_VGHIGH0 :
          value->rValue = model->B2vghigh0;
            return(OK);
        case  BSIM2_MOD_VGHIGHL :
          value->rValue = model->B2vghighL;
            return(OK);
        case  BSIM2_MOD_VGHIGHW :
          value->rValue = model->B2vghighW;
            return(OK);
        case  BSIM2_MOD_VGLOW0 :
          value->rValue = model->B2vglow0;
            return(OK);
        case  BSIM2_MOD_VGLOWL :
          value->rValue = model->B2vglowL;
            return(OK);
        case  BSIM2_MOD_VGLOWW :
          value->rValue = model->B2vglowW;
            return(OK);
        case  BSIM2_MOD_TOX :
          value->rValue = model->B2tox;
            return(OK);
        case  BSIM2_MOD_TEMP :
          value->rValue = model->B2temp;
            return(OK);
        case  BSIM2_MOD_VDD :
          value->rValue = model->B2vdd;
            return(OK);
        case  BSIM2_MOD_VGG :
          value->rValue = model->B2vgg;
            return(OK);
        case  BSIM2_MOD_VBB :
          value->rValue = model->B2vbb;
            return(OK);
        case BSIM2_MOD_CGSO:
            value->rValue = model->B2gateSourceOverlapCap; 
            return(OK);
        case BSIM2_MOD_CGDO:
            value->rValue = model->B2gateDrainOverlapCap; 
            return(OK);
        case BSIM2_MOD_CGBO:
            value->rValue = model->B2gateBulkOverlapCap; 
            return(OK);
        case BSIM2_MOD_XPART:
            value->iValue = model->B2channelChargePartitionFlag; 
            return(OK);
        case BSIM2_MOD_RSH:
            value->rValue = model->B2sheetResistance; 
            return(OK);
        case BSIM2_MOD_JS:
            value->rValue = model->B2jctSatCurDensity; 
            return(OK);
        case BSIM2_MOD_PB:
            value->rValue = model->B2bulkJctPotential; 
            return(OK);
        case BSIM2_MOD_MJ:
            value->rValue = model->B2bulkJctBotGradingCoeff; 
            return(OK);
        case BSIM2_MOD_PBSW:
            value->rValue = model->B2sidewallJctPotential; 
            return(OK);
        case BSIM2_MOD_MJSW:
            value->rValue = model->B2bulkJctSideGradingCoeff; 
            return(OK);
        case BSIM2_MOD_CJ:
            value->rValue = model->B2unitAreaJctCap; 
            return(OK);
        case BSIM2_MOD_CJSW:
            value->rValue = model->B2unitLengthSidewallJctCap; 
            return(OK);
        case BSIM2_MOD_DEFWIDTH:
            value->rValue = model->B2defaultWidth; 
            return(OK);
        case BSIM2_MOD_DELLENGTH:
            value->rValue = model->B2deltaLength; 
            return(OK);
        case BSIM2_MOD_AF:
            value->rValue = model->B2fNexp; 
            return(OK);
        case BSIM2_MOD_KF:
            value->rValue = model->B2fNcoef; 
            return(OK);
        default:
            return(E_BADPARM);
    }
    /* NOTREACHED */
}


