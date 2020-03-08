/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

#include "ngspice/ngspice.h"
#include "ngspice/const.h"
#include "ngspice/ifsim.h"
#include "jfetdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


/* ARGSUSED */
int
JFETparam(int param, IFvalue *value, GENinstance *inst, IFvalue *select)
{
    JFETinstance *here = (JFETinstance *)inst;

    NG_IGNORE(select);

    switch (param) {
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
            here->JFEToff = (value->iValue != 0);
            break;
        case JFET_IC:
            /* FALLTHROUGH added to suppress GCC warning due to
             * -Wimplicit-fallthrough flag */
            switch (value->v.numValue) {
                case 2:
                    here->JFETicVGS = *(value->v.vec.rVec+1);
                    here->JFETicVGSGiven = TRUE;
                    /* FALLTHROUGH */
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
