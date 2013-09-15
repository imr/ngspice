/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

#include "ngspice/ngspice.h"
#include "ngspice/ifsim.h"
#include "cccsdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


/* ARGSUSED */
int
CCCSparam(int param, IFvalue *value, GENinstance *inst, IFvalue *select)
{
    CCCSinstance *here = (CCCSinstance*)inst;

    NG_IGNORE(select);

    switch(param) {
        case CCCS_GAIN:
            here->CCCScoeff = value->rValue;
            if (here->CCCSmGiven)
                here->CCCScoeff *= here->CCCSmValue;
            here->CCCScoeffGiven = TRUE;
            break;
        case CCCS_CONTROL:
            here->CCCScontName = value->uValue;
            break;
        case CCCS_M:
            here->CCCSmValue = value->rValue;
            here->CCCSmGiven = TRUE;
            break;
        case CCCS_GAIN_SENS:
            here->CCCSsenParmNo = value->iValue;
            break;
        default:
            return(E_BADPARM);
    }
    return(OK);
}
