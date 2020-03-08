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
#include "mos2defs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"
#include "ngspice/fteext.h"


/* ARGSUSED */
int
MOS2param(int param, IFvalue *value, GENinstance *inst,
          IFvalue *select)
{
    double scale;

    MOS2instance *here = (MOS2instance *)inst;

    NG_IGNORE(select);

    if (!cp_getvar("scale", CP_REAL, &scale, 0))
        scale = 1;

    switch (param) {
        case MOS2_TEMP:
            here->MOS2temp = value->rValue+CONSTCtoK;
            here->MOS2tempGiven = TRUE;
            break;
        case MOS2_DTEMP:
            here->MOS2dtemp = value->rValue;
            here->MOS2dtempGiven = TRUE;
            break;
        case MOS2_M:
            here->MOS2m = value->rValue;
            here->MOS2mGiven = TRUE;
            break;   
        case MOS2_W:
            here->MOS2w = value->rValue * scale;
            here->MOS2wGiven = TRUE;
            break;
        case MOS2_L:
            here->MOS2l = value->rValue * scale;
            here->MOS2lGiven = TRUE;
            break;
        case MOS2_AS:
            here->MOS2sourceArea = value->rValue * scale * scale;
            here->MOS2sourceAreaGiven = TRUE;
            break;
        case MOS2_AD:
            here->MOS2drainArea = value->rValue * scale * scale;
            here->MOS2drainAreaGiven = TRUE;
            break;
        case MOS2_PS:
            here->MOS2sourcePerimiter = value->rValue * scale;
            here->MOS2sourcePerimiterGiven = TRUE;
            break;
        case MOS2_PD:
            here->MOS2drainPerimiter = value->rValue * scale;
            here->MOS2drainPerimiterGiven = TRUE;
            break;
        case MOS2_NRS:
            here->MOS2sourceSquares = value->rValue;
            here->MOS2sourceSquaresGiven = TRUE;
            break;
        case MOS2_NRD:
            here->MOS2drainSquares = value->rValue;
            here->MOS2drainSquaresGiven = TRUE;
            break;
        case MOS2_OFF:
            here->MOS2off = (value->iValue != 0);
            break;
        case MOS2_IC_VBS:
            here->MOS2icVBS = value->rValue;
            here->MOS2icVBSGiven = TRUE;
            break;
        case MOS2_IC_VDS:
            here->MOS2icVDS = value->rValue;
            here->MOS2icVDSGiven = TRUE;
            break;
        case MOS2_IC_VGS:
            here->MOS2icVGS = value->rValue;
            here->MOS2icVGSGiven = TRUE;
            break;
        case MOS2_IC:
            /* FALLTHROUGH added to suppress GCC warning due to
             * -Wimplicit-fallthrough flag */
            switch (value->v.numValue) {
                case 3:
                    here->MOS2icVBS = *(value->v.vec.rVec+2);
                    here->MOS2icVBSGiven = TRUE;
                    /* FALLTHROUGH */
                case 2:
                    here->MOS2icVGS = *(value->v.vec.rVec+1);
                    here->MOS2icVGSGiven = TRUE;
                    /* FALLTHROUGH */
                case 1:
                    here->MOS2icVDS = *(value->v.vec.rVec);
                    here->MOS2icVDSGiven = TRUE;
                    break;
                default:
                    return(E_BADPARM);
            }
            break;
        case MOS2_L_SENS:
            if(value->iValue) {
                here->MOS2senParmNo = 1;
                here->MOS2sens_l = 1;
            }
            break;
        case MOS2_W_SENS:
            if(value->iValue) {
                here->MOS2senParmNo = 1;
                here->MOS2sens_w = 1;
            }
            break;
        default:
            return(E_BADPARM);
    }
    return(OK);
}
