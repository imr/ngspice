/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: 2000 AlansFixes
**********/
/*
 */

#include "ngspice.h"
#include "const.h"
#include "ifsim.h"
#include "mos3defs.h"
#include "sperror.h"
#include "suffix.h"


/* ARGSUSED */
int
MOS3param(int param, IFvalue *value, GENinstance *inst, IFvalue *select)
{
    MOS3instance *here = (MOS3instance *)inst;
    switch(param) {
    	
        case MOS3_M:
            here->MOS3m = value->rValue;
            here->MOS3mGiven = TRUE;
            break;
        case MOS3_W:
            here->MOS3w = value->rValue;
            here->MOS3wGiven = TRUE;
            break;
        case MOS3_L:
            here->MOS3l = value->rValue;
            here->MOS3lGiven = TRUE;
            break;
        case MOS3_AS:
            here->MOS3sourceArea = value->rValue;
            here->MOS3sourceAreaGiven = TRUE;
            break;
        case MOS3_AD:
            here->MOS3drainArea = value->rValue;
            here->MOS3drainAreaGiven = TRUE;
            break;
        case MOS3_PS:
            here->MOS3sourcePerimiter = value->rValue;
            here->MOS3sourcePerimiterGiven = TRUE;
            break;
        case MOS3_PD:
            here->MOS3drainPerimiter = value->rValue;
            here->MOS3drainPerimiterGiven = TRUE;
            break;
        case MOS3_NRS:
            here->MOS3sourceSquares = value->rValue;
            here->MOS3sourceSquaresGiven = TRUE;
            break;
        case MOS3_NRD:
            here->MOS3drainSquares = value->rValue;
            here->MOS3drainSquaresGiven = TRUE;
            break;
        case MOS3_OFF:
            here->MOS3off = value->iValue;
            break;
        case MOS3_IC_VBS:
            here->MOS3icVBS = value->rValue;
            here->MOS3icVBSGiven = TRUE;
            break;
        case MOS3_IC_VDS:
            here->MOS3icVDS = value->rValue;
            here->MOS3icVDSGiven = TRUE;
            break;
        case MOS3_IC_VGS:
            here->MOS3icVGS = value->rValue;
            here->MOS3icVGSGiven = TRUE;
            break;
        case MOS3_TEMP:
            here->MOS3temp = value->rValue+CONSTCtoK;
            here->MOS3tempGiven = TRUE;
            break;
        case MOS3_DTEMP:
            here->MOS3dtemp = value->rValue;
            here->MOS3dtempGiven = TRUE;
            break;
        case MOS3_IC:
            switch(value->v.numValue){
                case 3:
                    here->MOS3icVBS = *(value->v.vec.rVec+2);
                    here->MOS3icVBSGiven = TRUE;
                case 2:
                    here->MOS3icVGS = *(value->v.vec.rVec+1);
                    here->MOS3icVGSGiven = TRUE;
                case 1:
                    here->MOS3icVDS = *(value->v.vec.rVec);
                    here->MOS3icVDSGiven = TRUE;
                    break;
                default:
                    return(E_BADPARM);
            }
            break;
        case MOS3_L_SENS:
            if(value->iValue) {
                here->MOS3senParmNo = 1;
                here->MOS3sens_l = 1;
            }
            break;
        case MOS3_W_SENS:
            if(value->iValue) {
                here->MOS3senParmNo = 1;
                here->MOS3sens_w = 1;
            }
            break;
        default:
            return(E_BADPARM);
    }
    return(OK);
}
