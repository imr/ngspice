/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Hong J. Park, Thomas L. Quarles
**********/
/*
 */

#include "ngspice/ngspice.h"
#include "bsim1def.h"
#include "ngspice/ifsim.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
B1mParam(int param, IFvalue *value, GENmodel *inMod)
{
    B1model *mod = (B1model*)inMod;
    switch(param) {
        case  BSIM1_MOD_VFB0 :
            mod->B1vfb0 = value->rValue;
            mod->B1vfb0Given = TRUE;
            break;
        case  BSIM1_MOD_VFBL :
            mod->B1vfbL = value->rValue;
            mod->B1vfbLGiven = TRUE;
            break;
        case  BSIM1_MOD_VFBW :
            mod->B1vfbW = value->rValue;
            mod->B1vfbWGiven = TRUE;
            break;
        case  BSIM1_MOD_PHI0 :
            mod->B1phi0 = value->rValue;
            mod->B1phi0Given = TRUE;
            break;
        case  BSIM1_MOD_PHIL :
            mod->B1phiL = value->rValue;
            mod->B1phiLGiven = TRUE;
            break;
        case  BSIM1_MOD_PHIW :
            mod->B1phiW = value->rValue;
            mod->B1phiWGiven = TRUE;
            break;
        case  BSIM1_MOD_K10 :
            mod->B1K10 = value->rValue;
            mod->B1K10Given = TRUE;
            break;
        case  BSIM1_MOD_K1L :
            mod->B1K1L = value->rValue;
            mod->B1K1LGiven = TRUE;
            break;
        case  BSIM1_MOD_K1W :
            mod->B1K1W = value->rValue;
            mod->B1K1WGiven = TRUE;
            break;
        case  BSIM1_MOD_K20 :
            mod->B1K20 = value->rValue;
            mod->B1K20Given = TRUE;
            break;
        case  BSIM1_MOD_K2L :
            mod->B1K2L = value->rValue;
            mod->B1K2LGiven = TRUE;
            break;
        case  BSIM1_MOD_K2W :
            mod->B1K2W = value->rValue;
            mod->B1K2WGiven = TRUE;
            break;
        case  BSIM1_MOD_ETA0 :
            mod->B1eta0 = value->rValue;
            mod->B1eta0Given = TRUE;
            break;
        case  BSIM1_MOD_ETAL :
            mod->B1etaL = value->rValue;
            mod->B1etaLGiven = TRUE;
            break;
        case  BSIM1_MOD_ETAW :
            mod->B1etaW = value->rValue;
            mod->B1etaWGiven = TRUE;
            break;
        case  BSIM1_MOD_ETAB0 :
            mod->B1etaB0 = value->rValue;
            mod->B1etaB0Given = TRUE;
            break;
        case  BSIM1_MOD_ETABL :
            mod->B1etaBl = value->rValue;
            mod->B1etaBlGiven = TRUE;
            break;
        case  BSIM1_MOD_ETABW :
            mod->B1etaBw = value->rValue;
            mod->B1etaBwGiven = TRUE;
            break;
        case  BSIM1_MOD_ETAD0 :
            mod->B1etaD0 = value->rValue;
            mod->B1etaD0Given = TRUE;
            break;
        case  BSIM1_MOD_ETADL :
            mod->B1etaDl = value->rValue;
            mod->B1etaDlGiven = TRUE;
            break;
        case  BSIM1_MOD_ETADW :
            mod->B1etaDw = value->rValue;
            mod->B1etaDwGiven = TRUE;
            break;
        case  BSIM1_MOD_DELTAL :
            mod->B1deltaL =  value->rValue;
            mod->B1deltaLGiven = TRUE;
            break;
        case  BSIM1_MOD_DELTAW :
            mod->B1deltaW =  value->rValue;
            mod->B1deltaWGiven = TRUE;
            break;
        case  BSIM1_MOD_MOBZERO :
            mod->B1mobZero = value->rValue;
            mod->B1mobZeroGiven = TRUE;
            break;
        case  BSIM1_MOD_MOBZEROB0 :
            mod->B1mobZeroB0 = value->rValue;
            mod->B1mobZeroB0Given = TRUE;
            break;
        case  BSIM1_MOD_MOBZEROBL :
            mod->B1mobZeroBl = value->rValue;
            mod->B1mobZeroBlGiven = TRUE;
            break;
        case  BSIM1_MOD_MOBZEROBW :
            mod->B1mobZeroBw = value->rValue;
            mod->B1mobZeroBwGiven = TRUE;
            break;
        case  BSIM1_MOD_MOBVDD0 :
            mod->B1mobVdd0 = value->rValue;
            mod->B1mobVdd0Given = TRUE;
            break;
        case  BSIM1_MOD_MOBVDDL :
            mod->B1mobVddl = value->rValue;
            mod->B1mobVddlGiven = TRUE;
            break;
        case  BSIM1_MOD_MOBVDDW :
            mod->B1mobVddw = value->rValue;
            mod->B1mobVddwGiven = TRUE;
            break;
        case  BSIM1_MOD_MOBVDDB0 :
            mod->B1mobVddB0 = value->rValue;
            mod->B1mobVddB0Given = TRUE;
            break;
        case  BSIM1_MOD_MOBVDDBL :
            mod->B1mobVddBl = value->rValue;
            mod->B1mobVddBlGiven = TRUE;
            break;
        case  BSIM1_MOD_MOBVDDBW :
            mod->B1mobVddBw = value->rValue;
            mod->B1mobVddBwGiven = TRUE;
            break;
        case  BSIM1_MOD_MOBVDDD0 :
            mod->B1mobVddD0 = value->rValue;
            mod->B1mobVddD0Given = TRUE;
            break;
        case  BSIM1_MOD_MOBVDDDL :
            mod->B1mobVddDl = value->rValue;
            mod->B1mobVddDlGiven = TRUE;
            break;
        case  BSIM1_MOD_MOBVDDDW :
            mod->B1mobVddDw = value->rValue;
            mod->B1mobVddDwGiven = TRUE;
            break;
        case  BSIM1_MOD_UGS0 :
            mod->B1ugs0 = value->rValue;
            mod->B1ugs0Given = TRUE;
            break;
        case  BSIM1_MOD_UGSL :
            mod->B1ugsL = value->rValue;
            mod->B1ugsLGiven = TRUE;
            break;
        case  BSIM1_MOD_UGSW :
            mod->B1ugsW = value->rValue;
            mod->B1ugsWGiven = TRUE;
            break;
        case  BSIM1_MOD_UGSB0 :
            mod->B1ugsB0 = value->rValue;
            mod->B1ugsB0Given = TRUE;
            break;
        case  BSIM1_MOD_UGSBL :
            mod->B1ugsBL = value->rValue;
            mod->B1ugsBLGiven = TRUE;
            break;
        case  BSIM1_MOD_UGSBW :
            mod->B1ugsBW = value->rValue;
            mod->B1ugsBWGiven = TRUE;
            break;
        case  BSIM1_MOD_UDS0 :
            mod->B1uds0 = value->rValue;
            mod->B1uds0Given = TRUE;
            break;
        case  BSIM1_MOD_UDSL :
            mod->B1udsL = value->rValue;
            mod->B1udsLGiven = TRUE;
            break;
        case  BSIM1_MOD_UDSW :
            mod->B1udsW = value->rValue;
            mod->B1udsWGiven = TRUE;
            break;
        case  BSIM1_MOD_UDSB0 :
            mod->B1udsB0 = value->rValue;
            mod->B1udsB0Given = TRUE;
            break;
        case  BSIM1_MOD_UDSBL :
            mod->B1udsBL = value->rValue;
            mod->B1udsBLGiven = TRUE;
            break;
        case  BSIM1_MOD_UDSBW :
            mod->B1udsBW = value->rValue;
            mod->B1udsBWGiven = TRUE;
            break;
        case  BSIM1_MOD_UDSD0 :
            mod->B1udsD0 = value->rValue;
            mod->B1udsD0Given = TRUE;
            break;
        case  BSIM1_MOD_UDSDL :
            mod->B1udsDL = value->rValue;
            mod->B1udsDLGiven = TRUE;
            break;
        case  BSIM1_MOD_UDSDW :
            mod->B1udsDW = value->rValue;
            mod->B1udsDWGiven = TRUE;
            break;
        case  BSIM1_MOD_N00 :
            mod->B1subthSlope0 = value->rValue;
            mod->B1subthSlope0Given = TRUE;
            break;
        case  BSIM1_MOD_N0L :
            mod->B1subthSlopeL = value->rValue;
            mod->B1subthSlopeLGiven = TRUE;
            break;
        case  BSIM1_MOD_N0W :
            mod->B1subthSlopeW = value->rValue;
            mod->B1subthSlopeWGiven = TRUE;
            break;
        case  BSIM1_MOD_NB0 :
            mod->B1subthSlopeB0 = value->rValue;
            mod->B1subthSlopeB0Given = TRUE;
            break;
        case  BSIM1_MOD_NBL :
            mod->B1subthSlopeBL = value->rValue;
            mod->B1subthSlopeBLGiven = TRUE;
            break;
        case  BSIM1_MOD_NBW :
            mod->B1subthSlopeBW = value->rValue;
            mod->B1subthSlopeBWGiven = TRUE;
            break;
        case  BSIM1_MOD_ND0 :
            mod->B1subthSlopeD0 = value->rValue;
            mod->B1subthSlopeD0Given = TRUE;
            break;
        case  BSIM1_MOD_NDL :
            mod->B1subthSlopeDL = value->rValue;
            mod->B1subthSlopeDLGiven = TRUE;
            break;
        case  BSIM1_MOD_NDW :
            mod->B1subthSlopeDW = value->rValue;
            mod->B1subthSlopeDWGiven = TRUE;
            break;
        case  BSIM1_MOD_TOX :
            mod->B1oxideThickness = value->rValue;
            mod->B1oxideThicknessGiven = TRUE;
            break;
        case  BSIM1_MOD_TEMP :
            mod->B1temp = value->rValue;
            mod->B1tempGiven = TRUE;
            break;
        case  BSIM1_MOD_VDD :
            mod->B1vdd = value->rValue;
            mod->B1vddGiven = TRUE;
            break;
        case  BSIM1_MOD_CGSO :
            mod->B1gateSourceOverlapCap = value->rValue;
            mod->B1gateSourceOverlapCapGiven = TRUE;
            break;
        case  BSIM1_MOD_CGDO :
            mod->B1gateDrainOverlapCap = value->rValue;
            mod->B1gateDrainOverlapCapGiven = TRUE;
            break;
        case  BSIM1_MOD_CGBO :
            mod->B1gateBulkOverlapCap = value->rValue;
            mod->B1gateBulkOverlapCapGiven = TRUE;
            break;
        case  BSIM1_MOD_XPART :
            mod->B1channelChargePartitionFlag = value->iValue ? 1 : 0;
            mod->B1channelChargePartitionFlagGiven = TRUE;
            break;
        case  BSIM1_MOD_RSH :
            mod->B1sheetResistance = value->rValue;
            mod->B1sheetResistanceGiven = TRUE;
            break;
        case  BSIM1_MOD_JS :
            mod->B1jctSatCurDensity = value->rValue;
            mod->B1jctSatCurDensityGiven = TRUE;
            break;
        case  BSIM1_MOD_PB :
            mod->B1bulkJctPotential = value->rValue;
            mod->B1bulkJctPotentialGiven = TRUE;
            break;
        case  BSIM1_MOD_MJ :
            mod->B1bulkJctBotGradingCoeff = value->rValue;
            mod->B1bulkJctBotGradingCoeffGiven = TRUE;
            break;
        case  BSIM1_MOD_PBSW :
            mod->B1sidewallJctPotential = value->rValue;
            mod->B1sidewallJctPotentialGiven = TRUE;
            break;
        case  BSIM1_MOD_MJSW :
            mod->B1bulkJctSideGradingCoeff = value->rValue;
            mod->B1bulkJctSideGradingCoeffGiven = TRUE;
            break;
        case  BSIM1_MOD_CJ :
            mod->B1unitAreaJctCap = value->rValue;
            mod->B1unitAreaJctCapGiven = TRUE;
            break;
        case  BSIM1_MOD_CJSW :
            mod->B1unitLengthSidewallJctCap = value->rValue;
            mod->B1unitLengthSidewallJctCapGiven = TRUE;
            break;
        case  BSIM1_MOD_DEFWIDTH :
            mod->B1defaultWidth = value->rValue;
            mod->B1defaultWidthGiven = TRUE;
            break;
        case  BSIM1_MOD_DELLENGTH :
            mod->B1deltaLength = value->rValue;
            mod->B1deltaLengthGiven = TRUE;
            break;
        case  BSIM1_MOD_AF :
            mod->B1fNexp = value->rValue;
            mod->B1fNexpGiven = TRUE;
            break;
        case  BSIM1_MOD_KF :
            mod->B1fNcoef = value->rValue;
            mod->B1fNcoefGiven = TRUE;
            break;
        case  BSIM1_MOD_NMOS  :
            if(value->iValue) {
                mod->B1type = 1;
                mod->B1typeGiven = TRUE;
            }
            break;
        case  BSIM1_MOD_PMOS  :
            if(value->iValue) {
                mod->B1type = - 1;
                mod->B1typeGiven = TRUE;
            }
            break;
        default:
            return(E_BADPARM);
    }
    return(OK);
}


