/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

#include "ngspice.h"
#include "ifsim.h"
#include "cccsdefs.h"
#include "sperror.h"
#include "suffix.h"


/* ARGSUSED */
int
CCCSparam(int param, IFvalue *value, GENinstance *inst, IFvalue *select)
{
    CCCSinstance *here = (CCCSinstance*)inst;
    switch(param) {
        case CCCS_GAIN:
            here->CCCScoeff = value->rValue;
            here->CCCScoeffGiven = TRUE;
            break;
        case CCCS_CONTROL:
            here->CCCScontName = value->uValue;
            break;
        case CCCS_GAIN_SENS:
            here->CCCSsenParmNo = value->iValue;
            break;
        default:
            return(E_BADPARM);
    }
    return(OK);
}
