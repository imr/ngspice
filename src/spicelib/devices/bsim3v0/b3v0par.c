/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1995 Min-Chie Jeng and Mansun Chan.
File: b3v0par.c
**********/

#include "ngspice/ngspice.h"
#include "ngspice/ifsim.h"
#include "bsim3v0def.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"
#include "ngspice/fteext.h"

int
BSIM3v0param(int param, IFvalue *value, GENinstance *inst, IFvalue *select)
{
    double scale;

    BSIM3v0instance *here = (BSIM3v0instance*)inst;

    NG_IGNORE(select);

    if (!cp_getvar("scale", CP_REAL, &scale, 0))
        scale = 1;

    switch (param) {
        case BSIM3v0_W:
            here->BSIM3v0w = value->rValue*scale;
            here->BSIM3v0wGiven = TRUE;
            break;
        case BSIM3v0_L:
            here->BSIM3v0l = value->rValue*scale;
            here->BSIM3v0lGiven = TRUE;
            break;
	case BSIM3v0_M:
            here->BSIM3v0m = value->rValue;
            here->BSIM3v0mGiven = TRUE;
            break;
        case BSIM3v0_AS:
            here->BSIM3v0sourceArea = value->rValue*scale*scale;
            here->BSIM3v0sourceAreaGiven = TRUE;
            break;
        case BSIM3v0_AD:
            here->BSIM3v0drainArea = value->rValue*scale*scale;
            here->BSIM3v0drainAreaGiven = TRUE;
            break;
        case BSIM3v0_PS:
            here->BSIM3v0sourcePerimeter = value->rValue*scale;
            here->BSIM3v0sourcePerimeterGiven = TRUE;
            break;
        case BSIM3v0_PD:
            here->BSIM3v0drainPerimeter = value->rValue*scale;
            here->BSIM3v0drainPerimeterGiven = TRUE;
            break;
        case BSIM3v0_NRS:
            here->BSIM3v0sourceSquares = value->rValue;
            here->BSIM3v0sourceSquaresGiven = TRUE;
            break;
        case BSIM3v0_NRD:
            here->BSIM3v0drainSquares = value->rValue;
            here->BSIM3v0drainSquaresGiven = TRUE;
            break;
        case BSIM3v0_OFF:
            here->BSIM3v0off = value->iValue;
            break;
        case BSIM3v0_IC_VBS:
            here->BSIM3v0icVBS = value->rValue;
            here->BSIM3v0icVBSGiven = TRUE;
            break;
        case BSIM3v0_IC_VDS:
            here->BSIM3v0icVDS = value->rValue;
            here->BSIM3v0icVDSGiven = TRUE;
            break;
        case BSIM3v0_IC_VGS:
            here->BSIM3v0icVGS = value->rValue;
            here->BSIM3v0icVGSGiven = TRUE;
            break;
        case BSIM3v0_NQSMOD:
            here->BSIM3v0nqsMod = value->iValue;
            here->BSIM3v0nqsModGiven = TRUE;
            break;
        case BSIM3v0_IC:
            /* FALLTHROUGH added to suppress GCC warning due to
             * -Wimplicit-fallthrough flag */
            switch (value->v.numValue) {
                case 3:
                    here->BSIM3v0icVBS = *(value->v.vec.rVec+2);
                    here->BSIM3v0icVBSGiven = TRUE;
                    /* FALLTHROUGH */
                case 2:
                    here->BSIM3v0icVGS = *(value->v.vec.rVec+1);
                    here->BSIM3v0icVGSGiven = TRUE;
                    /* FALLTHROUGH */
                case 1:
                    here->BSIM3v0icVDS = *(value->v.vec.rVec);
                    here->BSIM3v0icVDSGiven = TRUE;
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



