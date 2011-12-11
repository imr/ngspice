/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1988 Min-Chie Jeng, Hong J. Park, Thomas L. Quarles
**********/
/*
 */

#include "ngspice/ngspice.h"
#include "bsim2def.h"
#include "ngspice/ifsim.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
B2mParam(int param, IFvalue *value, GENmodel *inMod)
{
    B2model *mod = (B2model*)inMod;
    switch(param) {
        case  BSIM2_MOD_VFB0 :
            mod->B2vfb0 = value->rValue;
            mod->B2vfb0Given = TRUE;
            break;
        case  BSIM2_MOD_VFBL :
            mod->B2vfbL = value->rValue;
            mod->B2vfbLGiven = TRUE;
            break;
        case  BSIM2_MOD_VFBW :
            mod->B2vfbW = value->rValue;
            mod->B2vfbWGiven = TRUE;
            break;
        case  BSIM2_MOD_PHI0 :
            mod->B2phi0 = value->rValue;
            mod->B2phi0Given = TRUE;
            break;
        case  BSIM2_MOD_PHIL :
            mod->B2phiL = value->rValue;
            mod->B2phiLGiven = TRUE;
            break;
        case  BSIM2_MOD_PHIW :
            mod->B2phiW = value->rValue;
            mod->B2phiWGiven = TRUE;
            break;
        case  BSIM2_MOD_K10 :
            mod->B2k10 = value->rValue;
            mod->B2k10Given = TRUE;
            break;
        case  BSIM2_MOD_K1L :
            mod->B2k1L = value->rValue;
            mod->B2k1LGiven = TRUE;
            break;
        case  BSIM2_MOD_K1W :
            mod->B2k1W = value->rValue;
            mod->B2k1WGiven = TRUE;
            break;
        case  BSIM2_MOD_K20 :
            mod->B2k20 = value->rValue;
            mod->B2k20Given = TRUE;
            break;
        case  BSIM2_MOD_K2L :
            mod->B2k2L = value->rValue;
            mod->B2k2LGiven = TRUE;
            break;
        case  BSIM2_MOD_K2W :
            mod->B2k2W = value->rValue;
            mod->B2k2WGiven = TRUE;
            break;
        case  BSIM2_MOD_ETA00 :
            mod->B2eta00 = value->rValue;
            mod->B2eta00Given = TRUE;
            break;
        case  BSIM2_MOD_ETA0L :
            mod->B2eta0L = value->rValue;
            mod->B2eta0LGiven = TRUE;
            break;
        case  BSIM2_MOD_ETA0W :
            mod->B2eta0W = value->rValue;
            mod->B2eta0WGiven = TRUE;
            break;
        case  BSIM2_MOD_ETAB0 :
            mod->B2etaB0 = value->rValue;
            mod->B2etaB0Given = TRUE;
            break;
        case  BSIM2_MOD_ETABL :
            mod->B2etaBL = value->rValue;
            mod->B2etaBLGiven = TRUE;
            break;
        case  BSIM2_MOD_ETABW :
            mod->B2etaBW = value->rValue;
            mod->B2etaBWGiven = TRUE;
            break;
        case  BSIM2_MOD_DELTAL :
            mod->B2deltaL =  value->rValue;
            mod->B2deltaLGiven = TRUE;
            break;
        case  BSIM2_MOD_DELTAW :
            mod->B2deltaW =  value->rValue;
            mod->B2deltaWGiven = TRUE;
            break;
        case  BSIM2_MOD_MOB00 :
            mod->B2mob00 = value->rValue;
            mod->B2mob00Given = TRUE;
            break;
        case  BSIM2_MOD_MOB0B0 :
            mod->B2mob0B0 = value->rValue;
            mod->B2mob0B0Given = TRUE;
            break;
        case  BSIM2_MOD_MOB0BL :
            mod->B2mob0BL = value->rValue;
            mod->B2mob0BLGiven = TRUE;
            break;
        case  BSIM2_MOD_MOB0BW :
            mod->B2mob0BW = value->rValue;
            mod->B2mob0BWGiven = TRUE;
            break;
        case  BSIM2_MOD_MOBS00 :
            mod->B2mobs00 = value->rValue;
            mod->B2mobs00Given = TRUE;
            break;
        case  BSIM2_MOD_MOBS0L :
            mod->B2mobs0L = value->rValue;
            mod->B2mobs0LGiven = TRUE;
            break;
        case  BSIM2_MOD_MOBS0W :
            mod->B2mobs0W = value->rValue;
            mod->B2mobs0WGiven = TRUE;
            break;
        case  BSIM2_MOD_MOBSB0 :
            mod->B2mobsB0 = value->rValue;
            mod->B2mobsB0Given = TRUE;
            break;
        case  BSIM2_MOD_MOBSBL :
            mod->B2mobsBL = value->rValue;
            mod->B2mobsBLGiven = TRUE;
            break;
        case  BSIM2_MOD_MOBSBW :
            mod->B2mobsBW = value->rValue;
            mod->B2mobsBWGiven = TRUE;
            break;
        case  BSIM2_MOD_MOB200 :
            mod->B2mob200 = value->rValue;
            mod->B2mob200Given = TRUE;
            break;
        case  BSIM2_MOD_MOB20L :
            mod->B2mob20L = value->rValue;
            mod->B2mob20LGiven = TRUE;
            break;
        case  BSIM2_MOD_MOB20W :
            mod->B2mob20W = value->rValue;
            mod->B2mob20WGiven = TRUE;
            break;
        case  BSIM2_MOD_MOB2B0 :
            mod->B2mob2B0 = value->rValue;
            mod->B2mob2B0Given = TRUE;
            break;
        case  BSIM2_MOD_MOB2BL :
            mod->B2mob2BL = value->rValue;
            mod->B2mob2BLGiven = TRUE;
            break;
        case  BSIM2_MOD_MOB2BW :
            mod->B2mob2BW = value->rValue;
            mod->B2mob2BWGiven = TRUE;
            break;
        case  BSIM2_MOD_MOB2G0 :
            mod->B2mob2G0 = value->rValue;
            mod->B2mob2G0Given = TRUE;
            break;
        case  BSIM2_MOD_MOB2GL :
            mod->B2mob2GL = value->rValue;
            mod->B2mob2GLGiven = TRUE;
            break;
        case  BSIM2_MOD_MOB2GW :
            mod->B2mob2GW = value->rValue;
            mod->B2mob2GWGiven = TRUE;
            break;
        case  BSIM2_MOD_MOB300 :
            mod->B2mob300 = value->rValue;
            mod->B2mob300Given = TRUE;
            break;
        case  BSIM2_MOD_MOB30L :
            mod->B2mob30L = value->rValue;
            mod->B2mob30LGiven = TRUE;
            break;
        case  BSIM2_MOD_MOB30W :
            mod->B2mob30W = value->rValue;
            mod->B2mob30WGiven = TRUE;
            break;
        case  BSIM2_MOD_MOB3B0 :
            mod->B2mob3B0 = value->rValue;
            mod->B2mob3B0Given = TRUE;
            break;
        case  BSIM2_MOD_MOB3BL :
            mod->B2mob3BL = value->rValue;
            mod->B2mob3BLGiven = TRUE;
            break;
        case  BSIM2_MOD_MOB3BW :
            mod->B2mob3BW = value->rValue;
            mod->B2mob3BWGiven = TRUE;
            break;
        case  BSIM2_MOD_MOB3G0 :
            mod->B2mob3G0 = value->rValue;
            mod->B2mob3G0Given = TRUE;
            break;
        case  BSIM2_MOD_MOB3GL :
            mod->B2mob3GL = value->rValue;
            mod->B2mob3GLGiven = TRUE;
            break;
        case  BSIM2_MOD_MOB3GW :
            mod->B2mob3GW = value->rValue;
            mod->B2mob3GWGiven = TRUE;
            break;
        case  BSIM2_MOD_MOB400 :
            mod->B2mob400 = value->rValue;
            mod->B2mob400Given = TRUE;
            break;
        case  BSIM2_MOD_MOB40L :
            mod->B2mob40L = value->rValue;
            mod->B2mob40LGiven = TRUE;
            break;
        case  BSIM2_MOD_MOB40W :
            mod->B2mob40W = value->rValue;
            mod->B2mob40WGiven = TRUE;
            break;
        case  BSIM2_MOD_MOB4B0 :
            mod->B2mob4B0 = value->rValue;
            mod->B2mob4B0Given = TRUE;
            break;
        case  BSIM2_MOD_MOB4BL :
            mod->B2mob4BL = value->rValue;
            mod->B2mob4BLGiven = TRUE;
            break;
        case  BSIM2_MOD_MOB4BW :
            mod->B2mob4BW = value->rValue;
            mod->B2mob4BWGiven = TRUE;
            break;
        case  BSIM2_MOD_MOB4G0 :
            mod->B2mob4G0 = value->rValue;
            mod->B2mob4G0Given = TRUE;
            break;
        case  BSIM2_MOD_MOB4GL :
            mod->B2mob4GL = value->rValue;
            mod->B2mob4GLGiven = TRUE;
            break;
        case  BSIM2_MOD_MOB4GW :
            mod->B2mob4GW = value->rValue;
            mod->B2mob4GWGiven = TRUE;
            break;
        case  BSIM2_MOD_UA00 :
            mod->B2ua00 = value->rValue;
            mod->B2ua00Given = TRUE;
            break;
        case  BSIM2_MOD_UA0L :
            mod->B2ua0L = value->rValue;
            mod->B2ua0LGiven = TRUE;
            break;
        case  BSIM2_MOD_UA0W :
            mod->B2ua0W = value->rValue;
            mod->B2ua0WGiven = TRUE;
            break;
        case  BSIM2_MOD_UAB0 :
            mod->B2uaB0 = value->rValue;
            mod->B2uaB0Given = TRUE;
            break;
        case  BSIM2_MOD_UABL :
            mod->B2uaBL = value->rValue;
            mod->B2uaBLGiven = TRUE;
            break;
        case  BSIM2_MOD_UABW :
            mod->B2uaBW = value->rValue;
            mod->B2uaBWGiven = TRUE;
            break;
        case  BSIM2_MOD_UB00 :
            mod->B2ub00 = value->rValue;
            mod->B2ub00Given = TRUE;
            break;
        case  BSIM2_MOD_UB0L :
            mod->B2ub0L = value->rValue;
            mod->B2ub0LGiven = TRUE;
            break;
        case  BSIM2_MOD_UB0W :
            mod->B2ub0W = value->rValue;
            mod->B2ub0WGiven = TRUE;
            break;
        case  BSIM2_MOD_UBB0 :
            mod->B2ubB0 = value->rValue;
            mod->B2ubB0Given = TRUE;
            break;
        case  BSIM2_MOD_UBBL :
            mod->B2ubBL = value->rValue;
            mod->B2ubBLGiven = TRUE;
            break;
        case  BSIM2_MOD_UBBW :
            mod->B2ubBW = value->rValue;
            mod->B2ubBWGiven = TRUE;
            break;
        case  BSIM2_MOD_U100 :
            mod->B2u100 = value->rValue;
            mod->B2u100Given = TRUE;
            break;
        case  BSIM2_MOD_U10L :
            mod->B2u10L = value->rValue;
            mod->B2u10LGiven = TRUE;
            break;
        case  BSIM2_MOD_U10W :
            mod->B2u10W = value->rValue;
            mod->B2u10WGiven = TRUE;
            break;
        case  BSIM2_MOD_U1B0 :
            mod->B2u1B0 = value->rValue;
            mod->B2u1B0Given = TRUE;
            break;
        case  BSIM2_MOD_U1BL :
            mod->B2u1BL = value->rValue;
            mod->B2u1BLGiven = TRUE;
            break;
        case  BSIM2_MOD_U1BW :
            mod->B2u1BW = value->rValue;
            mod->B2u1BWGiven = TRUE;
            break;
        case  BSIM2_MOD_U1D0 :
            mod->B2u1D0 = value->rValue;
            mod->B2u1D0Given = TRUE;
            break;
        case  BSIM2_MOD_U1DL :
            mod->B2u1DL = value->rValue;
            mod->B2u1DLGiven = TRUE;
            break;
        case  BSIM2_MOD_U1DW :
            mod->B2u1DW = value->rValue;
            mod->B2u1DWGiven = TRUE;
            break;
        case  BSIM2_MOD_N00 :
            mod->B2n00 = value->rValue;
            mod->B2n00Given = TRUE;
            break;
        case  BSIM2_MOD_N0L :
            mod->B2n0L = value->rValue;
            mod->B2n0LGiven = TRUE;
            break;
        case  BSIM2_MOD_N0W :
            mod->B2n0W = value->rValue;
            mod->B2n0WGiven = TRUE;
            break;
        case  BSIM2_MOD_NB0 :
            mod->B2nB0 = value->rValue;
            mod->B2nB0Given = TRUE;
            break;
        case  BSIM2_MOD_NBL :
            mod->B2nBL = value->rValue;
            mod->B2nBLGiven = TRUE;
            break;
        case  BSIM2_MOD_NBW :
            mod->B2nBW = value->rValue;
            mod->B2nBWGiven = TRUE;
            break;
        case  BSIM2_MOD_ND0 :
            mod->B2nD0 = value->rValue;
            mod->B2nD0Given = TRUE;
            break;
        case  BSIM2_MOD_NDL :
            mod->B2nDL = value->rValue;
            mod->B2nDLGiven = TRUE;
            break;
        case  BSIM2_MOD_NDW :
            mod->B2nDW = value->rValue;
            mod->B2nDWGiven = TRUE;
            break;
        case  BSIM2_MOD_VOF00 :
            mod->B2vof00 = value->rValue;
            mod->B2vof00Given = TRUE;
            break;
        case  BSIM2_MOD_VOF0L :
            mod->B2vof0L = value->rValue;
            mod->B2vof0LGiven = TRUE;
            break;
        case  BSIM2_MOD_VOF0W :
            mod->B2vof0W = value->rValue;
            mod->B2vof0WGiven = TRUE;
            break;
        case  BSIM2_MOD_VOFB0 :
            mod->B2vofB0 = value->rValue;
            mod->B2vofB0Given = TRUE;
            break;
        case  BSIM2_MOD_VOFBL :
            mod->B2vofBL = value->rValue;
            mod->B2vofBLGiven = TRUE;
            break;
        case  BSIM2_MOD_VOFBW :
            mod->B2vofBW = value->rValue;
            mod->B2vofBWGiven = TRUE;
            break;
        case  BSIM2_MOD_VOFD0 :
            mod->B2vofD0 = value->rValue;
            mod->B2vofD0Given = TRUE;
            break;
        case  BSIM2_MOD_VOFDL :
            mod->B2vofDL = value->rValue;
            mod->B2vofDLGiven = TRUE;
            break;
        case  BSIM2_MOD_VOFDW :
            mod->B2vofDW = value->rValue;
            mod->B2vofDWGiven = TRUE;
            break;
        case  BSIM2_MOD_AI00 :
            mod->B2ai00 = value->rValue;
            mod->B2ai00Given = TRUE;
            break;
        case  BSIM2_MOD_AI0L :
            mod->B2ai0L = value->rValue;
            mod->B2ai0LGiven = TRUE;
            break;
        case  BSIM2_MOD_AI0W :
            mod->B2ai0W = value->rValue;
            mod->B2ai0WGiven = TRUE;
            break;
        case  BSIM2_MOD_AIB0 :
            mod->B2aiB0 = value->rValue;
            mod->B2aiB0Given = TRUE;
            break;
        case  BSIM2_MOD_AIBL :
            mod->B2aiBL = value->rValue;
            mod->B2aiBLGiven = TRUE;
            break;
        case  BSIM2_MOD_AIBW :
            mod->B2aiBW = value->rValue;
            mod->B2aiBWGiven = TRUE;
            break;
        case  BSIM2_MOD_BI00 :
            mod->B2bi00 = value->rValue;
            mod->B2bi00Given = TRUE;
            break;
        case  BSIM2_MOD_BI0L :
            mod->B2bi0L = value->rValue;
            mod->B2bi0LGiven = TRUE;
            break;
        case  BSIM2_MOD_BI0W :
            mod->B2bi0W = value->rValue;
            mod->B2bi0WGiven = TRUE;
            break;
        case  BSIM2_MOD_BIB0 :
            mod->B2biB0 = value->rValue;
            mod->B2biB0Given = TRUE;
            break;
        case  BSIM2_MOD_BIBL :
            mod->B2biBL = value->rValue;
            mod->B2biBLGiven = TRUE;
            break;
        case  BSIM2_MOD_BIBW :
            mod->B2biBW = value->rValue;
            mod->B2biBWGiven = TRUE;
            break;
        case  BSIM2_MOD_VGHIGH0 :
            mod->B2vghigh0 = value->rValue;
            mod->B2vghigh0Given = TRUE;
            break;
        case  BSIM2_MOD_VGHIGHL :
            mod->B2vghighL = value->rValue;
            mod->B2vghighLGiven = TRUE;
            break;
        case  BSIM2_MOD_VGHIGHW :
            mod->B2vghighW = value->rValue;
            mod->B2vghighWGiven = TRUE;
            break;
        case  BSIM2_MOD_VGLOW0 :
            mod->B2vglow0 = value->rValue;
            mod->B2vglow0Given = TRUE;
            break;
        case  BSIM2_MOD_VGLOWL :
            mod->B2vglowL = value->rValue;
            mod->B2vglowLGiven = TRUE;
            break;
        case  BSIM2_MOD_VGLOWW :
            mod->B2vglowW = value->rValue;
            mod->B2vglowWGiven = TRUE;
            break;
        case  BSIM2_MOD_TOX :
            mod->B2tox = value->rValue;
            mod->B2toxGiven = TRUE;
            break;
        case  BSIM2_MOD_TEMP :
            mod->B2temp = value->rValue;
            mod->B2tempGiven = TRUE;
            break;
        case  BSIM2_MOD_VDD :
            mod->B2vdd = value->rValue;
            mod->B2vddGiven = TRUE;
            break;
        case  BSIM2_MOD_VGG :
            mod->B2vgg = value->rValue;
            mod->B2vggGiven = TRUE;
            break;
        case  BSIM2_MOD_VBB :
            mod->B2vbb = value->rValue;
            mod->B2vbbGiven = TRUE;
            break;
        case  BSIM2_MOD_CGSO :
            mod->B2gateSourceOverlapCap = value->rValue;
            mod->B2gateSourceOverlapCapGiven = TRUE;
            break;
        case  BSIM2_MOD_CGDO :
            mod->B2gateDrainOverlapCap = value->rValue;
            mod->B2gateDrainOverlapCapGiven = TRUE;
            break;
        case  BSIM2_MOD_CGBO :
            mod->B2gateBulkOverlapCap = value->rValue;
            mod->B2gateBulkOverlapCapGiven = TRUE;
            break;
        case  BSIM2_MOD_XPART :
            mod->B2channelChargePartitionFlag = (value->iValue != 0);
            mod->B2channelChargePartitionFlagGiven = TRUE;
            break;
        case  BSIM2_MOD_RSH :
            mod->B2sheetResistance = value->rValue;
            mod->B2sheetResistanceGiven = TRUE;
            break;
        case  BSIM2_MOD_JS :
            mod->B2jctSatCurDensity = value->rValue;
            mod->B2jctSatCurDensityGiven = TRUE;
            break;
        case  BSIM2_MOD_PB :
            mod->B2bulkJctPotential = value->rValue;
            mod->B2bulkJctPotentialGiven = TRUE;
            break;
        case  BSIM2_MOD_MJ :
            mod->B2bulkJctBotGradingCoeff = value->rValue;
            mod->B2bulkJctBotGradingCoeffGiven = TRUE;
            break;
        case  BSIM2_MOD_PBSW :
            mod->B2sidewallJctPotential = value->rValue;
            mod->B2sidewallJctPotentialGiven = TRUE;
            break;
        case  BSIM2_MOD_MJSW :
            mod->B2bulkJctSideGradingCoeff = value->rValue;
            mod->B2bulkJctSideGradingCoeffGiven = TRUE;
            break;
        case  BSIM2_MOD_CJ :
            mod->B2unitAreaJctCap = value->rValue;
            mod->B2unitAreaJctCapGiven = TRUE;
            break;
        case  BSIM2_MOD_CJSW :
            mod->B2unitLengthSidewallJctCap = value->rValue;
            mod->B2unitLengthSidewallJctCapGiven = TRUE;
            break;
        case  BSIM2_MOD_DEFWIDTH :
            mod->B2defaultWidth = value->rValue;
            mod->B2defaultWidthGiven = TRUE;
            break;
        case  BSIM2_MOD_DELLENGTH :
            mod->B2deltaLength = value->rValue;
            mod->B2deltaLengthGiven = TRUE;
            break;
        case  BSIM2_MOD_AF :
            mod->B2fNexp = value->rValue;
            mod->B2fNexpGiven = TRUE;
            break;
        case  BSIM2_MOD_KF :
            mod->B2fNcoef = value->rValue;
            mod->B2fNcoefGiven = TRUE;
            break;
        case  BSIM2_MOD_NMOS  :
            if(value->iValue) {
                mod->B2type = 1;
                mod->B2typeGiven = TRUE;
            }
            break;
        case  BSIM2_MOD_PMOS  :
            if(value->iValue) {
                mod->B2type = - 1;
                mod->B2typeGiven = TRUE;
            }
            break;
        default:
            return(E_BADPARM);
    }
    return(OK);
}
