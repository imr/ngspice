/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: 2000 AlansFixes
**********/
/*
 */

#include "ngspice/ngspice.h"
#include "ngspice/const.h"
#include "ngspice/ifsim.h"
#include "mos3defs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"
#include "ngspice/fteext.h"


/* ARGSUSED */
int
MOS3param(int param, IFvalue *value, GENinstance *inst, IFvalue *select)
{
    double scale;

    MOS3instance *here = (MOS3instance *)inst;

    NG_IGNORE(select);

    if (!cp_getvar("scale", CP_REAL, &scale, 0))
        scale = 1;

    switch (param) {
        case MOS3_M:
            here->MOS3m = value->rValue;
            here->MOS3mGiven = TRUE;
            break;
        case MOS3_W:
            here->MOS3w = value->rValue * scale;
            here->MOS3wGiven = TRUE;
            break;
        case MOS3_L:
            here->MOS3l = value->rValue * scale;
            here->MOS3lGiven = TRUE;
            break;
        case MOS3_AS:
            here->MOS3sourceArea = value->rValue * scale * scale;
            here->MOS3sourceAreaGiven = TRUE;
            break;
        case MOS3_AD:
            here->MOS3drainArea = value->rValue * scale * scale;
            here->MOS3drainAreaGiven = TRUE;
            break;
        case MOS3_PS:
            here->MOS3sourcePerimiter = value->rValue * scale;
            here->MOS3sourcePerimiterGiven = TRUE;
            break;
        case MOS3_PD:
            here->MOS3drainPerimiter = value->rValue * scale;
            here->MOS3drainPerimiterGiven = TRUE;
            break;
        case MOS3_NRS:
            here->MOS3sourceSquares = value->rValue;
            here->MOS3sourceSquaresGiven = TRUE;
            break;
        case MOS3_NRD:
            here->MOS3drainSquares = value->rValue;
            here->MOS3drainSquaresGiven = TRUE;
            break;
        case MOS3_OFF:
            here->MOS3off = (value->iValue != 0);
            break;
        case MOS3_IC_VBS:
            here->MOS3icVBS = value->rValue;
            here->MOS3icVBSGiven = TRUE;
            break;
        case MOS3_IC_VDS:
            here->MOS3icVDS = value->rValue;
            here->MOS3icVDSGiven = TRUE;
            break;
        case MOS3_IC_VGS:
            here->MOS3icVGS = value->rValue;
            here->MOS3icVGSGiven = TRUE;
            break;
        case MOS3_TEMP:
            here->MOS3temp = value->rValue+CONSTCtoK;
            here->MOS3tempGiven = TRUE;
            break;
        case MOS3_DTEMP:
            here->MOS3dtemp = value->rValue;
            here->MOS3dtempGiven = TRUE;
            break;
        case MOS3_IC:
            /* FALLTHROUGH added to suppress GCC warning due to
             * -Wimplicit-fallthrough flag */
            switch (value->v.numValue) {
                case 3:
                    here->MOS3icVBS = *(value->v.vec.rVec+2);
                    here->MOS3icVBSGiven = TRUE;
                    /* FALLTHROUGH */
                case 2:
                    here->MOS3icVGS = *(value->v.vec.rVec+1);
                    here->MOS3icVGSGiven = TRUE;
                    /* FALLTHROUGH */
                case 1:
                    here->MOS3icVDS = *(value->v.vec.rVec);
                    here->MOS3icVDSGiven = TRUE;
                    break;
                default:
                    return(E_BADPARM);
            }
            break;
        case MOS3_L_SENS:
            if(value->iValue) {
                here->MOS3senParmNo = 1;
                here->MOS3sens_l = 1;
            }
            break;
        case MOS3_W_SENS:
            if(value->iValue) {
                here->MOS3senParmNo = 1;
                here->MOS3sens_w = 1;
            }
            break;
        default:
            return(E_BADPARM);
    }
    return(OK);
}
