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
#include "mos1defs.h"
#include "sperror.h"
#include "suffix.h"


/* ARGSUSED */
int
MOS1param(int param, IFvalue *value, GENinstance *inst, IFvalue *select)
{
    MOS1instance *here = (MOS1instance *)inst;
    switch(param) {
        case MOS1_TEMP:
            here->MOS1temp = value->rValue+CONSTCtoK;
            here->MOS1tempGiven = TRUE;
            break;
        case MOS1_DTEMP:
            here->MOS1dtemp = value->rValue;
            here->MOS1dtempGiven = TRUE;
            break;
        case MOS1_M:
            here->MOS1m = value->rValue;
            here->MOS1mGiven = TRUE;
            break;
        case MOS1_W:
            here->MOS1w = value->rValue;
            here->MOS1wGiven = TRUE;
            break;
        case MOS1_L:
            here->MOS1l = value->rValue;
            here->MOS1lGiven = TRUE;
            break;
        case MOS1_AS:
            here->MOS1sourceArea = value->rValue;
            here->MOS1sourceAreaGiven = TRUE;
            break;
        case MOS1_AD:
            here->MOS1drainArea = value->rValue;
            here->MOS1drainAreaGiven = TRUE;
            break;
        case MOS1_PS:
            here->MOS1sourcePerimiter = value->rValue;
            here->MOS1sourcePerimiterGiven = TRUE;
            break;
        case MOS1_PD:
            here->MOS1drainPerimiter = value->rValue;
            here->MOS1drainPerimiterGiven = TRUE;
            break;
        case MOS1_NRS:
            here->MOS1sourceSquares = value->rValue;
            here->MOS1sourceSquaresGiven = TRUE;
            break;
        case MOS1_NRD:
            here->MOS1drainSquares = value->rValue;
            here->MOS1drainSquaresGiven = TRUE;
            break;
        case MOS1_OFF:
            here->MOS1off = value->iValue;
            break;
        case MOS1_IC_VBS:
            here->MOS1icVBS = value->rValue;
            here->MOS1icVBSGiven = TRUE;
            break;
        case MOS1_IC_VDS:
            here->MOS1icVDS = value->rValue;
            here->MOS1icVDSGiven = TRUE;
            break;
        case MOS1_IC_VGS:
            here->MOS1icVGS = value->rValue;
            here->MOS1icVGSGiven = TRUE;
            break;
        case MOS1_IC:
            switch(value->v.numValue){
                case 3:
                    here->MOS1icVBS = *(value->v.vec.rVec+2);
                    here->MOS1icVBSGiven = TRUE;
                case 2:
                    here->MOS1icVGS = *(value->v.vec.rVec+1);
                    here->MOS1icVGSGiven = TRUE;
                case 1:
                    here->MOS1icVDS = *(value->v.vec.rVec);
                    here->MOS1icVDSGiven = TRUE;
                    break;
                default:
                    return(E_BADPARM);
            }
            break;
        case MOS1_L_SENS:
            if(value->iValue) {
                here->MOS1senParmNo = 1;
                here->MOS1sens_l = 1;
            }
            break;
        case MOS1_W_SENS:
            if(value->iValue) {
                here->MOS1senParmNo = 1;
                here->MOS1sens_w = 1;
            }
            break;
        default:
            return(E_BADPARM);
    }
    return(OK);
}
