/**********
 * Copyright 1990 Regents of the University of California. All rights reserved.
 * File: b3v1par.c
 * Author: 1995 Min-Chie Jeng and Mansun Chan. 
 * Modified by Paolo Nenzi 2002
 **********/
 
/* 
 * Release Notes: 
 * BSIM3v3.1,   Released by yuhua  96/12/08
 */

#include "ngspice.h"
#include "ifsim.h"
#include "bsim3v1def.h"
#include "sperror.h"
#include "suffix.h"

int
BSIM3v1param(int param, IFvalue *value, GENinstance *inst, IFvalue *select)
{
    BSIM3v1instance *here = (BSIM3v1instance*)inst;
    switch(param) 
    {   case BSIM3v1_W:
            here->BSIM3v1w = value->rValue;
            here->BSIM3v1wGiven = TRUE;
            break;
        case BSIM3v1_L:
            here->BSIM3v1l = value->rValue;
            here->BSIM3v1lGiven = TRUE;
            break;
	    case BSIM3v1_M:
            here->BSIM3v1m = value->rValue;
            here->BSIM3v1mGiven = TRUE;
            break;
        case BSIM3v1_AS:
            here->BSIM3v1sourceArea = value->rValue;
            here->BSIM3v1sourceAreaGiven = TRUE;
            break;
        case BSIM3v1_AD:
            here->BSIM3v1drainArea = value->rValue;
            here->BSIM3v1drainAreaGiven = TRUE;
            break;
        case BSIM3v1_PS:
            here->BSIM3v1sourcePerimeter = value->rValue;
            here->BSIM3v1sourcePerimeterGiven = TRUE;
            break;
        case BSIM3v1_PD:
            here->BSIM3v1drainPerimeter = value->rValue;
            here->BSIM3v1drainPerimeterGiven = TRUE;
            break;
        case BSIM3v1_NRS:
            here->BSIM3v1sourceSquares = value->rValue;
            here->BSIM3v1sourceSquaresGiven = TRUE;
            break;
        case BSIM3v1_NRD:
            here->BSIM3v1drainSquares = value->rValue;
            here->BSIM3v1drainSquaresGiven = TRUE;
            break;
        case BSIM3v1_OFF:
            here->BSIM3v1off = value->iValue;
            break;
        case BSIM3v1_IC_VBS:
            here->BSIM3v1icVBS = value->rValue;
            here->BSIM3v1icVBSGiven = TRUE;
            break;
        case BSIM3v1_IC_VDS:
            here->BSIM3v1icVDS = value->rValue;
            here->BSIM3v1icVDSGiven = TRUE;
            break;
        case BSIM3v1_IC_VGS:
            here->BSIM3v1icVGS = value->rValue;
            here->BSIM3v1icVGSGiven = TRUE;
            break;
        case BSIM3v1_NQSMOD:
            here->BSIM3v1nqsMod = value->iValue;
            here->BSIM3v1nqsModGiven = TRUE;
            break;
        case BSIM3v1_IC:
            switch(value->v.numValue){
                case 3:
                    here->BSIM3v1icVBS = *(value->v.vec.rVec+2);
                    here->BSIM3v1icVBSGiven = TRUE;
                case 2:
                    here->BSIM3v1icVGS = *(value->v.vec.rVec+1);
                    here->BSIM3v1icVGSGiven = TRUE;
                case 1:
                    here->BSIM3v1icVDS = *(value->v.vec.rVec);
                    here->BSIM3v1icVDSGiven = TRUE;
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



