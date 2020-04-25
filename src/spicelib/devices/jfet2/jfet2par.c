/**********
based on jfetpar.c
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles

Modified to jfet2 for PS model definition ( Anthony E. Parker )
   Copyright 1994  Macquarie University, Sydney Australia.
**********/
/*
 */

#include "ngspice/ngspice.h"
#include "ngspice/const.h"
#include "ngspice/ifsim.h"
#include "jfet2defs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


/* ARGSUSED */
int
JFET2param(int param, IFvalue *value, GENinstance *inst, IFvalue *select)
{
    JFET2instance *here = (JFET2instance *)inst;

    NG_IGNORE(select);

    switch (param) {
        case JFET2_TEMP:
            here->JFET2temp = value->rValue+CONSTCtoK;
            here->JFET2tempGiven = TRUE;
            break;
        case JFET2_DTEMP:
            here->JFET2temp = value->rValue;
            here->JFET2tempGiven = TRUE;
            break;
        case JFET2_AREA:
            here->JFET2area = value->rValue;
            here->JFET2areaGiven = TRUE;
            break;
         case JFET2_M:
            here->JFET2m = value->rValue;
            here->JFET2mGiven = TRUE;
            break;
        case JFET2_IC_VDS:
            here->JFET2icVDS = value->rValue;
            here->JFET2icVDSGiven = TRUE;
            break;
        case JFET2_IC_VGS:
            here->JFET2icVGS = value->rValue;
            here->JFET2icVGSGiven = TRUE;
            break;
        case JFET2_OFF:
            here->JFET2off = (value->iValue != 0);
            break;
        case JFET2_IC:
            /* FALLTHROUGH added to suppress GCC warning due to
             * -Wimplicit-fallthrough flag */
            switch (value->v.numValue) {
                case 2:
                    here->JFET2icVGS = *(value->v.vec.rVec+1);
                    here->JFET2icVGSGiven = TRUE;
                    /* FALLTHROUGH */
                case 1:
                    here->JFET2icVDS = *(value->v.vec.rVec);
                    here->JFET2icVDSGiven = TRUE;
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
