/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1995 Min-Chie Jeng and Mansun Chan.
Modified by Paolo Nenzi 2002
File: b3v1spar.c
**********/

#include "ngspice.h"
#include "ifsim.h"
#include "bsim3v1sdef.h"
#include "sperror.h"
#include "suffix.h"

int
BSIM3v1Sparam(int param, IFvalue *value, GENinstance *inst, IFvalue *select)
{
    BSIM3v1Sinstance *here = (BSIM3v1Sinstance*)inst;
    switch(param) 
    {   case BSIM3v1S_W:
            here->BSIM3v1Sw = value->rValue;
            here->BSIM3v1SwGiven = TRUE;
            break;
        case BSIM3v1S_L:
            here->BSIM3v1Sl = value->rValue;
            here->BSIM3v1SlGiven = TRUE;
            break;
        case BSIM3v1S_AS:
            here->BSIM3v1SsourceArea = value->rValue;
            here->BSIM3v1SsourceAreaGiven = TRUE;
            break;
        case BSIM3v1S_AD:
            here->BSIM3v1SdrainArea = value->rValue;
            here->BSIM3v1SdrainAreaGiven = TRUE;
            break;
        case BSIM3v1S_PS:
            here->BSIM3v1SsourcePerimeter = value->rValue;
            here->BSIM3v1SsourcePerimeterGiven = TRUE;
            break;
        case BSIM3v1S_PD:
            here->BSIM3v1SdrainPerimeter = value->rValue;
            here->BSIM3v1SdrainPerimeterGiven = TRUE;
            break;
        case BSIM3v1S_NRS:
            here->BSIM3v1SsourceSquares = value->rValue;
            here->BSIM3v1SsourceSquaresGiven = TRUE;
            break;
        case BSIM3v1S_NRD:
            here->BSIM3v1SdrainSquares = value->rValue;
            here->BSIM3v1SdrainSquaresGiven = TRUE;
            break;
        case BSIM3v1S_OFF:
            here->BSIM3v1Soff = value->iValue;
            break;
        case BSIM3v1S_M:
            here->BSIM3v1Sm = value->rValue;
            break;
        case BSIM3v1S_IC_VBS:
            here->BSIM3v1SicVBS = value->rValue;
            here->BSIM3v1SicVBSGiven = TRUE;
            break;
        case BSIM3v1S_IC_VDS:
            here->BSIM3v1SicVDS = value->rValue;
            here->BSIM3v1SicVDSGiven = TRUE;
            break;
        case BSIM3v1S_IC_VGS:
            here->BSIM3v1SicVGS = value->rValue;
            here->BSIM3v1SicVGSGiven = TRUE;
            break;
        case BSIM3v1S_NQSMOD:
            here->BSIM3v1SnqsMod = value->iValue;
            here->BSIM3v1SnqsModGiven = TRUE;
            break;
        case BSIM3v1S_IC:
            switch(value->v.numValue){
                case 3:
                    here->BSIM3v1SicVBS = *(value->v.vec.rVec+2);
                    here->BSIM3v1SicVBSGiven = TRUE;
                case 2:
                    here->BSIM3v1SicVGS = *(value->v.vec.rVec+1);
                    here->BSIM3v1SicVGSGiven = TRUE;
                case 1:
                    here->BSIM3v1SicVDS = *(value->v.vec.rVec);
                    here->BSIM3v1SicVDSGiven = TRUE;
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



