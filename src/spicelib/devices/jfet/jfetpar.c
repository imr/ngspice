/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

#include "ngspice.h"
#include "const.h"
#include "ifsim.h"
#include "jfetdefs.h"
#include "sperror.h"
#include "suffix.h"


/* ARGSUSED */
int
JFETparam(int param, IFvalue *value, GENinstance *inst, IFvalue *select)
{
    JFETinstance *here = (JFETinstance *)inst;
    switch(param) {
        case JFET_TEMP:
            here->JFETtemp = value->rValue+CONSTCtoK;
            here->JFETtempGiven = TRUE;
            break;
        case JFET_DTEMP:
            here->JFETdtemp = value->rValue;
            here->JFETdtempGiven = TRUE;
            break;
        case JFET_AREA:
            here->JFETarea = value->rValue;
            here->JFETareaGiven = TRUE;
            break;
       case JFET_M:
            here->JFETm = value->rValue;
            here->JFETmGiven = TRUE;
            break;
        case JFET_IC_VDS:
            here->JFETicVDS = value->rValue;
            here->JFETicVDSGiven = TRUE;
            break;
        case JFET_IC_VGS:
            here->JFETicVGS = value->rValue;
            here->JFETicVGSGiven = TRUE;
            break;
        case JFET_OFF:
            here->JFEToff = value->iValue;
            break;
        case JFET_IC:
            switch(value->v.numValue) {
                case 2:
                    here->JFETicVGS = *(value->v.vec.rVec+1);
                    here->JFETicVGSGiven = TRUE;
                case 1:
                    here->JFETicVDS = *(value->v.vec.rVec);
                    here->JFETicVDSGiven = TRUE;
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
