/* $Id$  */
/*
 $Log$
 Revision 1.1  2000-04-27 20:03:59  pnenzi
 Initial revision

 * Revision 3.2 1998/6/16  18:00:00  Weidong 
 * BSIM3v3.2 release
 *
*/
static char rcsid[] = "$Id$";

/*************************************/

/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1995 Min-Chie Jeng and Mansun Chan.
File: b3v2par.c
**********/

#include "ngspice.h"
#include <stdio.h>
#include "ifsim.h"
#include "bsim3v2def.h"
#include "sperror.h"
#include "suffix.h"

int
BSIM3V2param(param,value,inst,select)
int param;
IFvalue *value;
GENinstance *inst;
IFvalue *select;
{
    BSIM3V2instance *here = (BSIM3V2instance*)inst;
    switch(param) 
    {   case BSIM3V2_W:
            here->BSIM3V2w = value->rValue;
            here->BSIM3V2wGiven = TRUE;
            break;
        case BSIM3V2_L:
            here->BSIM3V2l = value->rValue;
            here->BSIM3V2lGiven = TRUE;
            break;
        case BSIM3V2_AS:
            here->BSIM3V2sourceArea = value->rValue;
            here->BSIM3V2sourceAreaGiven = TRUE;
            break;
        case BSIM3V2_AD:
            here->BSIM3V2drainArea = value->rValue;
            here->BSIM3V2drainAreaGiven = TRUE;
            break;
        case BSIM3V2_PS:
            here->BSIM3V2sourcePerimeter = value->rValue;
            here->BSIM3V2sourcePerimeterGiven = TRUE;
            break;
        case BSIM3V2_PD:
            here->BSIM3V2drainPerimeter = value->rValue;
            here->BSIM3V2drainPerimeterGiven = TRUE;
            break;
        case BSIM3V2_NRS:
            here->BSIM3V2sourceSquares = value->rValue;
            here->BSIM3V2sourceSquaresGiven = TRUE;
            break;
        case BSIM3V2_NRD:
            here->BSIM3V2drainSquares = value->rValue;
            here->BSIM3V2drainSquaresGiven = TRUE;
            break;
        case BSIM3V2_OFF:
            here->BSIM3V2off = value->iValue;
            break;
        case BSIM3V2_IC_VBS:
            here->BSIM3V2icVBS = value->rValue;
            here->BSIM3V2icVBSGiven = TRUE;
            break;
        case BSIM3V2_IC_VDS:
            here->BSIM3V2icVDS = value->rValue;
            here->BSIM3V2icVDSGiven = TRUE;
            break;
        case BSIM3V2_IC_VGS:
            here->BSIM3V2icVGS = value->rValue;
            here->BSIM3V2icVGSGiven = TRUE;
            break;
        case BSIM3V2_NQSMOD:
            here->BSIM3V2nqsMod = value->iValue;
            here->BSIM3V2nqsModGiven = TRUE;
            break;
        case BSIM3V2_IC:
            switch(value->v.numValue){
                case 3:
                    here->BSIM3V2icVBS = *(value->v.vec.rVec+2);
                    here->BSIM3V2icVBSGiven = TRUE;
                case 2:
                    here->BSIM3V2icVGS = *(value->v.vec.rVec+1);
                    here->BSIM3V2icVGSGiven = TRUE;
                case 1:
                    here->BSIM3V2icVDS = *(value->v.vec.rVec);
                    here->BSIM3V2icVDSGiven = TRUE;
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



