/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1995 Min-Chie Jeng and Mansun Chan.
File: b3v1par.c
**********/

#include "ngspice.h"
#include <stdio.h>
#include "ifsim.h"
#include "bsim3v1def.h"
#include "sperror.h"
#include "suffix.h"

int
BSIM3V1param(param,value,inst,select)
int param;
IFvalue *value;
GENinstance *inst;
IFvalue *select;
{
    BSIM3V1instance *here = (BSIM3V1instance*)inst;
    switch(param) 
    {   case BSIM3V1_W:
            here->BSIM3V1w = value->rValue;
            here->BSIM3V1wGiven = TRUE;
            break;
        case BSIM3V1_L:
            here->BSIM3V1l = value->rValue;
            here->BSIM3V1lGiven = TRUE;
            break;
        case BSIM3V1_AS:
            here->BSIM3V1sourceArea = value->rValue;
            here->BSIM3V1sourceAreaGiven = TRUE;
            break;
        case BSIM3V1_AD:
            here->BSIM3V1drainArea = value->rValue;
            here->BSIM3V1drainAreaGiven = TRUE;
            break;
        case BSIM3V1_PS:
            here->BSIM3V1sourcePerimeter = value->rValue;
            here->BSIM3V1sourcePerimeterGiven = TRUE;
            break;
        case BSIM3V1_PD:
            here->BSIM3V1drainPerimeter = value->rValue;
            here->BSIM3V1drainPerimeterGiven = TRUE;
            break;
        case BSIM3V1_NRS:
            here->BSIM3V1sourceSquares = value->rValue;
            here->BSIM3V1sourceSquaresGiven = TRUE;
            break;
        case BSIM3V1_NRD:
            here->BSIM3V1drainSquares = value->rValue;
            here->BSIM3V1drainSquaresGiven = TRUE;
            break;
        case BSIM3V1_OFF:
            here->BSIM3V1off = value->iValue;
            break;
        case BSIM3V1_M:
            here->BSIM3V1m = value->rValue;
            break;
        case BSIM3V1_IC_VBS:
            here->BSIM3V1icVBS = value->rValue;
            here->BSIM3V1icVBSGiven = TRUE;
            break;
        case BSIM3V1_IC_VDS:
            here->BSIM3V1icVDS = value->rValue;
            here->BSIM3V1icVDSGiven = TRUE;
            break;
        case BSIM3V1_IC_VGS:
            here->BSIM3V1icVGS = value->rValue;
            here->BSIM3V1icVGSGiven = TRUE;
            break;
        case BSIM3V1_NQSMOD:
            here->BSIM3V1nqsMod = value->iValue;
            here->BSIM3V1nqsModGiven = TRUE;
            break;
        case BSIM3V1_IC:
            switch(value->v.numValue){
                case 3:
                    here->BSIM3V1icVBS = *(value->v.vec.rVec+2);
                    here->BSIM3V1icVBSGiven = TRUE;
                case 2:
                    here->BSIM3V1icVGS = *(value->v.vec.rVec+1);
                    here->BSIM3V1icVGSGiven = TRUE;
                case 1:
                    here->BSIM3V1icVDS = *(value->v.vec.rVec);
                    here->BSIM3V1icVDSGiven = TRUE;
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



