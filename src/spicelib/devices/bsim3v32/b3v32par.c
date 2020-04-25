/**** BSIM3v3.2.4, Released by Xuemei Xi 12/21/2001 ****/

/**********
 * Copyright 2001 Regents of the University of California. All rights reserved.
 * File: b3par.c of BSIM3v3.2.4
 * Author: 1995 Min-Chie Jeng and Mansun Chan
 * Author: 1997-1999 Weidong Liu.
 * Author: 2001 Xuemei Xi
 * Modified by Paolo Nenzi 2002 and Dietmar Warning 2003
 **********/

#include "ngspice/ngspice.h"
#include "ngspice/ifsim.h"
#include "bsim3v32def.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"
#include "ngspice/fteext.h"

int
BSIM3v32param (int param, IFvalue *value, GENinstance *inst, IFvalue *select)
{
    double scale;

    BSIM3v32instance *here = (BSIM3v32instance*)inst;

    NG_IGNORE(select);

    if (!cp_getvar("scale", CP_REAL, &scale, 0))
        scale = 1;

    switch (param) {
        case BSIM3v32_W:
            here->BSIM3v32w = value->rValue*scale;
            here->BSIM3v32wGiven = TRUE;
            break;
        case BSIM3v32_L:
            here->BSIM3v32l = value->rValue*scale;
            here->BSIM3v32lGiven = TRUE;
            break;
        case BSIM3v32_M:
            here->BSIM3v32m = value->rValue;
            here->BSIM3v32mGiven = TRUE;
            break;
        case BSIM3v32_AS:
            here->BSIM3v32sourceArea = value->rValue*scale*scale;
            here->BSIM3v32sourceAreaGiven = TRUE;
            break;
        case BSIM3v32_AD:
            here->BSIM3v32drainArea = value->rValue*scale*scale;
            here->BSIM3v32drainAreaGiven = TRUE;
            break;
        case BSIM3v32_PS:
            here->BSIM3v32sourcePerimeter = value->rValue*scale;
            here->BSIM3v32sourcePerimeterGiven = TRUE;
            break;
        case BSIM3v32_PD:
            here->BSIM3v32drainPerimeter = value->rValue*scale;
            here->BSIM3v32drainPerimeterGiven = TRUE;
            break;
        case BSIM3v32_NRS:
            here->BSIM3v32sourceSquares = value->rValue;
            here->BSIM3v32sourceSquaresGiven = TRUE;
            break;
        case BSIM3v32_NRD:
            here->BSIM3v32drainSquares = value->rValue;
            here->BSIM3v32drainSquaresGiven = TRUE;
            break;
        case BSIM3v32_OFF:
            here->BSIM3v32off = value->iValue;
            break;
        case BSIM3v32_IC_VBS:
            here->BSIM3v32icVBS = value->rValue;
            here->BSIM3v32icVBSGiven = TRUE;
            break;
        case BSIM3v32_IC_VDS:
            here->BSIM3v32icVDS = value->rValue;
            here->BSIM3v32icVDSGiven = TRUE;
            break;
        case BSIM3v32_IC_VGS:
            here->BSIM3v32icVGS = value->rValue;
            here->BSIM3v32icVGSGiven = TRUE;
            break;
        case BSIM3v32_NQSMOD:
            here->BSIM3v32nqsMod = value->iValue;
            here->BSIM3v32nqsModGiven = TRUE;
            break;
        case BSIM3v32_GEO:
            here->BSIM3v32geo = value->iValue;
            here->BSIM3v32geoGiven = TRUE;
            break;
        case BSIM3v32_DELVTO:
            here->BSIM3v32delvto = value->rValue;
            here->BSIM3v32delvtoGiven = TRUE;
            break;
        case BSIM3v32_MULU0:
            here->BSIM3v32mulu0 = value->rValue;
            here->BSIM3v32mulu0Given = TRUE;
            break;
        case BSIM3v32_IC:
            /* FALLTHROUGH added to suppress GCC warning due to
             * -Wimplicit-fallthrough flag */
            switch (value->v.numValue) {
                case 3:
                    here->BSIM3v32icVBS = *(value->v.vec.rVec+2);
                    here->BSIM3v32icVBSGiven = TRUE;
                    /* FALLTHROUGH */
                case 2:
                    here->BSIM3v32icVGS = *(value->v.vec.rVec+1);
                    here->BSIM3v32icVGSGiven = TRUE;
                    /* FALLTHROUGH */
                case 1:
                    here->BSIM3v32icVDS = *(value->v.vec.rVec);
                    here->BSIM3v32icVDSGiven = TRUE;
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

