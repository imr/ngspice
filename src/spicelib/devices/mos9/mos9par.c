/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: Alan Gillespie
**********/
/*
 */

#include "ngspice/ngspice.h"
#include "ngspice/const.h"
#include "ngspice/ifsim.h"
#include "mos9defs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


/* ARGSUSED */
int
MOS9param(int param, IFvalue *value, GENinstance *inst,
          IFvalue *select)
{
    MOS9instance *here = (MOS9instance *)inst;

    NG_IGNORE(select);

    switch (param) {
        case MOS9_M:
            here->MOS9m = value->rValue;
            here->MOS9mGiven = TRUE;
            break;
        case MOS9_W:
            here->MOS9w = value->rValue;
            here->MOS9wGiven = TRUE;
            break;
        case MOS9_L:
            here->MOS9l = value->rValue;
            here->MOS9lGiven = TRUE;
            break;
        case MOS9_AS:
            here->MOS9sourceArea = value->rValue;
            here->MOS9sourceAreaGiven = TRUE;
            break;
        case MOS9_AD:
            here->MOS9drainArea = value->rValue;
            here->MOS9drainAreaGiven = TRUE;
            break;
        case MOS9_PS:
            here->MOS9sourcePerimiter = value->rValue;
            here->MOS9sourcePerimiterGiven = TRUE;
            break;
        case MOS9_PD:
            here->MOS9drainPerimiter = value->rValue;
            here->MOS9drainPerimiterGiven = TRUE;
            break;
        case MOS9_NRS:
            here->MOS9sourceSquares = value->rValue;
            here->MOS9sourceSquaresGiven = TRUE;
            break;
        case MOS9_NRD:
            here->MOS9drainSquares = value->rValue;
            here->MOS9drainSquaresGiven = TRUE;
            break;
        case MOS9_OFF:
            here->MOS9off = (value->iValue != 0);
            break;
        case MOS9_IC_VBS:
            here->MOS9icVBS = value->rValue;
            here->MOS9icVBSGiven = TRUE;
            break;
        case MOS9_IC_VDS:
            here->MOS9icVDS = value->rValue;
            here->MOS9icVDSGiven = TRUE;
            break;
        case MOS9_IC_VGS:
            here->MOS9icVGS = value->rValue;
            here->MOS9icVGSGiven = TRUE;
            break;
        case MOS9_TEMP:
            here->MOS9temp = value->rValue+CONSTCtoK;
            here->MOS9tempGiven = TRUE;
            break;
        case MOS9_DTEMP:
            here->MOS9dtemp = value->rValue;
            here->MOS9dtempGiven = TRUE;
            break;
        case MOS9_IC:
            /* FALLTHROUGH added to suppress GCC warning due to
             * -Wimplicit-fallthrough flag */
            switch (value->v.numValue) {
                case 3:
                    here->MOS9icVBS = *(value->v.vec.rVec+2);
                    here->MOS9icVBSGiven = TRUE;
                    /* FALLTHROUGH */
                case 2:
                    here->MOS9icVGS = *(value->v.vec.rVec+1);
                    here->MOS9icVGSGiven = TRUE;
                    /* FALLTHROUGH */
                case 1:
                    here->MOS9icVDS = *(value->v.vec.rVec);
                    here->MOS9icVDSGiven = TRUE;
                    break;
                default:
                    return(E_BADPARM);
            }
            break;
        case MOS9_L_SENS:
            if(value->iValue) {
                here->MOS9senParmNo = 1;
                here->MOS9sens_l = 1;
            }
            break;
        case MOS9_W_SENS:
            if(value->iValue) {
                here->MOS9senParmNo = 1;
                here->MOS9sens_w = 1;
            }
            break;
        default:
            return(E_BADPARM);
    }
    return(OK);
}
