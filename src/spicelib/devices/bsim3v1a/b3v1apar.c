/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1995 Min-Chie Jeng and Mansun Chan.
Modified by Paolo Nenzi 2002
File: b3v1apar.c
**********/

#include "ngspice.h"
#include "ifsim.h"
#include "bsim3v1adef.h"
#include "sperror.h"
#include "suffix.h"

int
BSIM3v1Aparam(int param, IFvalue *value, GENinstance *inst, IFvalue *select)
{
    BSIM3v1Ainstance *here = (BSIM3v1Ainstance*)inst;
    switch(param) 
    {   case BSIM3v1A_W:
            here->BSIM3v1Aw = value->rValue;
            here->BSIM3v1AwGiven = TRUE;
            break;
        case BSIM3v1A_L:
            here->BSIM3v1Al = value->rValue;
            here->BSIM3v1AlGiven = TRUE;
            break;
        case BSIM3v1A_M:
            here->BSIM3v1Am = value->rValue;
            here->BSIM3v1AmGiven = TRUE;
            break;
        case BSIM3v1A_AS:
            here->BSIM3v1AsourceArea = value->rValue;
            here->BSIM3v1AsourceAreaGiven = TRUE;
            break;
        case BSIM3v1A_AD:
            here->BSIM3v1AdrainArea = value->rValue;
            here->BSIM3v1AdrainAreaGiven = TRUE;
            break;
        case BSIM3v1A_PS:
            here->BSIM3v1AsourcePerimeter = value->rValue;
            here->BSIM3v1AsourcePerimeterGiven = TRUE;
            break;
        case BSIM3v1A_PD:
            here->BSIM3v1AdrainPerimeter = value->rValue;
            here->BSIM3v1AdrainPerimeterGiven = TRUE;
            break;
        case BSIM3v1A_NRS:
            here->BSIM3v1AsourceSquares = value->rValue;
            here->BSIM3v1AsourceSquaresGiven = TRUE;
            break;
        case BSIM3v1A_NRD:
            here->BSIM3v1AdrainSquares = value->rValue;
            here->BSIM3v1AdrainSquaresGiven = TRUE;
            break;
        case BSIM3v1A_OFF:
            here->BSIM3v1Aoff = value->iValue;
            break;
        case BSIM3v1A_IC_VBS:
            here->BSIM3v1AicVBS = value->rValue;
            here->BSIM3v1AicVBSGiven = TRUE;
            break;
        case BSIM3v1A_IC_VDS:
            here->BSIM3v1AicVDS = value->rValue;
            here->BSIM3v1AicVDSGiven = TRUE;
            break;
        case BSIM3v1A_IC_VGS:
            here->BSIM3v1AicVGS = value->rValue;
            here->BSIM3v1AicVGSGiven = TRUE;
            break;
        case BSIM3v1A_NQSMOD:
            here->BSIM3v1AnqsMod = value->iValue;
            here->BSIM3v1AnqsModGiven = TRUE;
            break;
        case BSIM3v1A_IC:
            switch(value->v.numValue){
                case 3:
                    here->BSIM3v1AicVBS = *(value->v.vec.rVec+2);
                    here->BSIM3v1AicVBSGiven = TRUE;
                case 2:
                    here->BSIM3v1AicVGS = *(value->v.vec.rVec+1);
                    here->BSIM3v1AicVGSGiven = TRUE;
                case 1:
                    here->BSIM3v1AicVDS = *(value->v.vec.rVec);
                    here->BSIM3v1AicVDSGiven = TRUE;
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



