/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Hong J. Park, Thomas L. Quarles
**********/
/*
 */

#include "ngspice.h"
#include "ifsim.h"
#include "bsim1def.h"
#include "sperror.h"
#include "suffix.h"


/* ARGSUSED */
int
B1param(int param, IFvalue *value, GENinstance *inst, 
        IFvalue *select)
{
    B1instance *here = (B1instance*)inst;
    switch(param) {
        case BSIM1_W:
            here->B1w = value->rValue;
            here->B1wGiven = TRUE;
            break;
        case BSIM1_L:
            here->B1l = value->rValue;
            here->B1lGiven = TRUE;
            break;
        case BSIM1_M:
            here->B1m = value->rValue;
            here->B1mGiven = TRUE;
            break;
        case BSIM1_AS:
            here->B1sourceArea = value->rValue;
            here->B1sourceAreaGiven = TRUE;
            break;
        case BSIM1_AD:
            here->B1drainArea = value->rValue;
            here->B1drainAreaGiven = TRUE;
            break;
        case BSIM1_PS:
            here->B1sourcePerimeter = value->rValue;
            here->B1sourcePerimeterGiven = TRUE;
            break;
        case BSIM1_PD:
            here->B1drainPerimeter = value->rValue;
            here->B1drainPerimeterGiven = TRUE;
            break;
        case BSIM1_NRS:
            here->B1sourceSquares = value->rValue;
            here->B1sourceSquaresGiven = TRUE;
            break;
        case BSIM1_NRD:
            here->B1drainSquares = value->rValue;
            here->B1drainSquaresGiven = TRUE;
            break;
        case BSIM1_OFF:
            here->B1off = value->iValue;
            break;
        case BSIM1_IC_VBS:
            here->B1icVBS = value->rValue;
            here->B1icVBSGiven = TRUE;
            break;
        case BSIM1_IC_VDS:
            here->B1icVDS = value->rValue;
            here->B1icVDSGiven = TRUE;
            break;
        case BSIM1_IC_VGS:
            here->B1icVGS = value->rValue;
            here->B1icVGSGiven = TRUE;
            break;
        case BSIM1_IC:
            switch(value->v.numValue){
                case 3:
                    here->B1icVBS = *(value->v.vec.rVec+2);
                    here->B1icVBSGiven = TRUE;
                case 2:
                    here->B1icVGS = *(value->v.vec.rVec+1);
                    here->B1icVGSGiven = TRUE;
                case 1:
                    here->B1icVDS = *(value->v.vec.rVec);
                    here->B1icVDSGiven = TRUE;
                    break;
                default:
                    return(E_BADPARM);
            }
            break;
        default:
            return(E_BADPARM);
    }
    return(OK);
}


