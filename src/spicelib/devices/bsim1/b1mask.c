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
#include "bsim1def.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


/*ARGSUSED*/
int
B1mAsk(CKTcircuit *ckt, GENmodel *inst, int which, IFvalue *value)
{
    B1model *model = (B1model *)inst;

    NG_IGNORE(ckt);

        switch(which) {
        case BSIM1_MOD_VFB0: 
            value->rValue = model->B1vfb0; 
            return(OK);
        case BSIM1_MOD_VFBL:
            value->rValue = model->B1vfbL; 
            return(OK);
        case BSIM1_MOD_VFBW:
            value->rValue = model->B1vfbW; 
            return(OK);
        case BSIM1_MOD_PHI0:
            value->rValue = model->B1phi0; 
            return(OK);
        case BSIM1_MOD_PHIL:
            value->rValue = model->B1phiL; 
            return(OK);
        case BSIM1_MOD_PHIW:
            value->rValue = model->B1phiW; 
            return(OK);
        case BSIM1_MOD_K10:
            value->rValue = model->B1K10; 
            return(OK);
        case BSIM1_MOD_K1L:
            value->rValue = model->B1K1L; 
            return(OK);
        case BSIM1_MOD_K1W:
            value->rValue = model->B1K1W; 
            return(OK);
        case BSIM1_MOD_K20:
            value->rValue = model->B1K20; 
            return(OK);
        case BSIM1_MOD_K2L:
            value->rValue = model->B1K2L; 
            return(OK);
        case BSIM1_MOD_K2W:
            value->rValue = model->B1K2W; 
            return(OK);
        case BSIM1_MOD_ETA0:
            value->rValue = model->B1eta0; 
            return(OK);
        case BSIM1_MOD_ETAL:
            value->rValue = model->B1etaL; 
            return(OK);
        case BSIM1_MOD_ETAW:
            value->rValue = model->B1etaW; 
            return(OK);
        case BSIM1_MOD_ETAB0:
            value->rValue = model->B1etaB0; 
            return(OK);
        case BSIM1_MOD_ETABL:
            value->rValue = model->B1etaBl; 
            return(OK);
        case BSIM1_MOD_ETABW:
            value->rValue = model->B1etaBw; 
            return(OK);
        case BSIM1_MOD_ETAD0:
            value->rValue = model->B1etaD0; 
            return(OK);
        case BSIM1_MOD_ETADL:
            value->rValue = model->B1etaDl; 
            return(OK);
        case BSIM1_MOD_ETADW:
            value->rValue = model->B1etaDw; 
            return(OK);
        case BSIM1_MOD_DELTAL:
            value->rValue = model->B1deltaL; 
            return(OK);
        case BSIM1_MOD_DELTAW:
            value->rValue = model->B1deltaW; 
            return(OK);
        case BSIM1_MOD_MOBZERO:
            value->rValue = model->B1mobZero; 
            return(OK);
        case BSIM1_MOD_MOBZEROB0:
            value->rValue = model->B1mobZeroB0; 
            return(OK);
        case BSIM1_MOD_MOBZEROBL:
            value->rValue = model->B1mobZeroBl; 
            return(OK);
        case BSIM1_MOD_MOBZEROBW:
            value->rValue = model->B1mobZeroBw; 
            return(OK);
        case BSIM1_MOD_MOBVDD0:
            value->rValue = model->B1mobVdd0; 
            return(OK);
        case BSIM1_MOD_MOBVDDL:
            value->rValue = model->B1mobVddl; 
            return(OK);
        case BSIM1_MOD_MOBVDDW:
            value->rValue = model->B1mobVddw; 
            return(OK);
        case BSIM1_MOD_MOBVDDB0:
            value->rValue = model->B1mobVddB0; 
            return(OK);
        case BSIM1_MOD_MOBVDDBL:
            value->rValue = model->B1mobVddBl; 
            return(OK);
        case BSIM1_MOD_MOBVDDBW:
            value->rValue = model->B1mobVddBw; 
            return(OK);
        case BSIM1_MOD_MOBVDDD0:
            value->rValue = model->B1mobVddD0; 
            return(OK);
        case BSIM1_MOD_MOBVDDDL:
            value->rValue = model->B1mobVddDl; 
            return(OK);
        case BSIM1_MOD_MOBVDDDW:
            value->rValue = model->B1mobVddDw; 
            return(OK);
        case BSIM1_MOD_UGS0:
            value->rValue = model->B1ugs0; 
            return(OK);
        case BSIM1_MOD_UGSL:
            value->rValue = model->B1ugsL; 
            return(OK);
        case BSIM1_MOD_UGSW:
            value->rValue = model->B1ugsW; 
            return(OK);
        case BSIM1_MOD_UGSB0:
            value->rValue = model->B1ugsB0; 
            return(OK);
        case BSIM1_MOD_UGSBL:
            value->rValue = model->B1ugsBL; 
            return(OK);
        case BSIM1_MOD_UGSBW:
            value->rValue = model->B1ugsBW; 
            return(OK);
        case BSIM1_MOD_UDS0:
            value->rValue = model->B1uds0; 
            return(OK);
        case BSIM1_MOD_UDSL:
            value->rValue = model->B1udsL; 
            return(OK);
        case BSIM1_MOD_UDSW:
            value->rValue = model->B1udsW; 
            return(OK);
        case BSIM1_MOD_UDSB0:
            value->rValue = model->B1udsB0; 
            return(OK);
        case BSIM1_MOD_UDSBL:
            value->rValue = model->B1udsBL; 
            return(OK);
        case BSIM1_MOD_UDSBW:
            value->rValue = model->B1udsBW; 
            return(OK);
        case BSIM1_MOD_UDSD0:
            value->rValue = model->B1udsD0; 
            return(OK);
        case BSIM1_MOD_UDSDL:
            value->rValue = model->B1udsDL; 
            return(OK);
        case BSIM1_MOD_UDSDW:
            value->rValue = model->B1udsDW; 
            return(OK);
        case BSIM1_MOD_N00:
            value->rValue = model->B1subthSlope0; 
            return(OK);
        case BSIM1_MOD_N0L:
            value->rValue = model->B1subthSlopeL; 
            return(OK);
        case BSIM1_MOD_N0W:
            value->rValue = model->B1subthSlopeW; 
            return(OK);
        case BSIM1_MOD_NB0:
            value->rValue = model->B1subthSlopeB0; 
            return(OK);
        case BSIM1_MOD_NBL:
            value->rValue = model->B1subthSlopeBL; 
            return(OK);
        case BSIM1_MOD_NBW:
            value->rValue = model->B1subthSlopeBW; 
            return(OK);
        case BSIM1_MOD_ND0:
            value->rValue = model->B1subthSlopeD0; 
            return(OK);
        case BSIM1_MOD_NDL:
            value->rValue = model->B1subthSlopeDL; 
            return(OK);
        case BSIM1_MOD_NDW:
            value->rValue = model->B1subthSlopeDW; 
            return(OK);
        case BSIM1_MOD_TOX:
            value->rValue = model->B1oxideThickness; 
            return(OK);
        case BSIM1_MOD_TEMP:
            value->rValue = model->B1temp; 
            return(OK);
        case BSIM1_MOD_VDD:
            value->rValue = model->B1vdd; 
            return(OK);
        case BSIM1_MOD_CGSO:
            value->rValue = model->B1gateSourceOverlapCap; 
            return(OK);
        case BSIM1_MOD_CGDO:
            value->rValue = model->B1gateDrainOverlapCap; 
            return(OK);
        case BSIM1_MOD_CGBO:
            value->rValue = model->B1gateBulkOverlapCap; 
            return(OK);
        case BSIM1_MOD_XPART:
            value->iValue = model->B1channelChargePartitionFlag; 
            return(OK);
        case BSIM1_MOD_RSH:
            value->rValue = model->B1sheetResistance; 
            return(OK);
        case BSIM1_MOD_JS:
            value->rValue = model->B1jctSatCurDensity; 
            return(OK);
        case BSIM1_MOD_PB:
            value->rValue = model->B1bulkJctPotential; 
            return(OK);
        case BSIM1_MOD_MJ:
            value->rValue = model->B1bulkJctBotGradingCoeff; 
            return(OK);
        case BSIM1_MOD_PBSW:
            value->rValue = model->B1sidewallJctPotential; 
            return(OK);
        case BSIM1_MOD_MJSW:
            value->rValue = model->B1bulkJctSideGradingCoeff; 
            return(OK);
        case BSIM1_MOD_CJ:
            value->rValue = model->B1unitAreaJctCap; 
            return(OK);
        case BSIM1_MOD_CJSW:
            value->rValue = model->B1unitLengthSidewallJctCap; 
            return(OK);
        case BSIM1_MOD_DEFWIDTH:
            value->rValue = model->B1defaultWidth; 
            return(OK);
        case BSIM1_MOD_DELLENGTH:
            value->rValue = model->B1deltaLength; 
            return(OK);
        case BSIM1_MOD_AF:
            value->rValue = model->B1fNexp; 
            return(OK);
        case BSIM1_MOD_KF:
            value->rValue = model->B1fNcoef; 
            return(OK);
        default:
            return(E_BADPARM);
    }
    /* NOTREACHED */
}

