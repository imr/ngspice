/**********
Copyright 1993: T. Ytterdal, K. Lee, M. Shur and T. A. Fjeldly. All rights reserved.
Author: Trond Ytterdal
**********/

#include "ngspice.h"
#include "const.h"
#include "ifsim.h"
#include "mesadefs.h"
#include "sperror.h"
#include "suffix.h"


/* ARGSUSED */
int
MESAparam(int param, IFvalue *value, GENinstance *inst, IFvalue *select)
{
    MESAinstance *here = (MESAinstance*)inst;
    switch(param) {
        case MESA_LENGTH:
            here->MESAlength = value->rValue;
            here->MESAlengthGiven = TRUE;
            break;
        case MESA_WIDTH:
            here->MESAwidth = value->rValue;
            here->MESAwidthGiven = TRUE;
            break;
        case MESA_M:
            here->MESAm = value->rValue;
            here->MESAmGiven = TRUE;
            break;

        case MESA_IC_VDS:
            here->MESAicVDS = value->rValue;
            here->MESAicVDSGiven = TRUE;
            break;
        case MESA_IC_VGS:
            here->MESAicVGS = value->rValue;
            here->MESAicVGSGiven = TRUE;
            break;
        case MESA_OFF:
            here->MESAoff = value->iValue;
            break;
        case MESA_IC:
            switch(value->v.numValue) {
                case 2:
                    here->MESAicVGS = *(value->v.vec.rVec+1);
                    here->MESAicVGSGiven = TRUE;
                case 1:
                    here->MESAicVDS = *(value->v.vec.rVec);
                    here->MESAicVDSGiven = TRUE;
                    break;
                default:
                    return(E_BADPARM);
            }
            break;
        case MESA_TD:
            here->MESAtd = value->rValue+CONSTCtoK;
            here->MESAtdGiven = TRUE;
            break;
        case MESA_TS:
            here->MESAts = value->rValue+CONSTCtoK;
            here->MESAtsGiven = TRUE;
            break;
        case MESA_DTEMP:
            here->MESAdtemp = value->rValue;
            here->MESAdtempGiven = TRUE;
            break;
        default:
            return(E_BADPARM);
    }
    return(OK);
}
