/**** BSIM3v3.3.0, Released by Xuemei Xi 07/29/2005 ****/

/**********
 * Copyright 2004 Regents of the University of California. All rights reserved.
 * File: b3par.c of BSIM3v3.3.0
 * Author: 1995 Min-Chie Jeng and Mansun Chan
 * Author: 1997-1999 Weidong Liu.
 * Author: 2001 Xuemei Xi
 **********/

#include "ngspice/ngspice.h"
#include "ngspice/ifsim.h"
#include "bsim3def.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"
#include "ngspice/fteext.h"

int
BSIM3param (
int param,
IFvalue *value,
GENinstance *inst,
IFvalue *select)
{
    double scale;

    BSIM3instance *here = (BSIM3instance*)inst;

    NG_IGNORE(select);

    if (!cp_getvar("scale", CP_REAL, &scale, 0))
        scale = 1;

    switch (param) {
        case BSIM3_W:
            here->BSIM3w = value->rValue*scale;
            here->BSIM3wGiven = TRUE;
            break;
        case BSIM3_L:
            here->BSIM3l = value->rValue*scale;
            here->BSIM3lGiven = TRUE;
            break;
        case BSIM3_M:
            here->BSIM3m = value->rValue;
            here->BSIM3mGiven = TRUE;
            break;
        case BSIM3_AS:
            here->BSIM3sourceArea = value->rValue*scale*scale;
            here->BSIM3sourceAreaGiven = TRUE;
            break;
        case BSIM3_AD:
            here->BSIM3drainArea = value->rValue*scale*scale;
            here->BSIM3drainAreaGiven = TRUE;
            break;
        case BSIM3_PS:
            here->BSIM3sourcePerimeter = value->rValue*scale;
            here->BSIM3sourcePerimeterGiven = TRUE;
            break;
        case BSIM3_PD:
            here->BSIM3drainPerimeter = value->rValue*scale;
            here->BSIM3drainPerimeterGiven = TRUE;
            break;
        case BSIM3_NRS:
            here->BSIM3sourceSquares = value->rValue;
            here->BSIM3sourceSquaresGiven = TRUE;
            break;
        case BSIM3_NRD:
            here->BSIM3drainSquares = value->rValue;
            here->BSIM3drainSquaresGiven = TRUE;
            break;
        case BSIM3_OFF:
            here->BSIM3off = value->iValue;
            break;
        case BSIM3_IC_VBS:
            here->BSIM3icVBS = value->rValue;
            here->BSIM3icVBSGiven = TRUE;
            break;
        case BSIM3_IC_VDS:
            here->BSIM3icVDS = value->rValue;
            here->BSIM3icVDSGiven = TRUE;
            break;
        case BSIM3_IC_VGS:
            here->BSIM3icVGS = value->rValue;
            here->BSIM3icVGSGiven = TRUE;
            break;
        case BSIM3_NQSMOD:
            here->BSIM3nqsMod = value->iValue;
            here->BSIM3nqsModGiven = TRUE;
            break;
        case BSIM3_ACNQSMOD:
            here->BSIM3acnqsMod = value->iValue;
            here->BSIM3acnqsModGiven = TRUE;
            break;
        case BSIM3_GEO:
            here->BSIM3geo = value->iValue;
            here->BSIM3geoGiven = TRUE;
            break;
        case BSIM3_DELVTO:
            here->BSIM3delvto = value->rValue;
            here->BSIM3delvtoGiven = TRUE;
            break;
        case BSIM3_MULU0:
            here->BSIM3mulu0 = value->rValue;
            here->BSIM3mulu0Given = TRUE;
            break;
        case BSIM3_IC:
            /* FALLTHROUGH added to suppress GCC warning due to
             * -Wimplicit-fallthrough flag */
            switch (value->v.numValue) {
                case 3:
                    here->BSIM3icVBS = *(value->v.vec.rVec+2);
                    here->BSIM3icVBSGiven = TRUE;
                    /* FALLTHROUGH */
                case 2:
                    here->BSIM3icVGS = *(value->v.vec.rVec+1);
                    here->BSIM3icVGSGiven = TRUE;
                    /* FALLTHROUGH */
                case 1:
                    here->BSIM3icVDS = *(value->v.vec.rVec);
                    here->BSIM3icVDSGiven = TRUE;
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



