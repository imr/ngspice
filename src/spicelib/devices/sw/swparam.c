/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Gordon Jacobs
**********/

#include "ngspice/ngspice.h"
#include "swdefs.h"
#include "ngspice/ifsim.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
SWparam(int param, IFvalue *value, GENinstance *inst, IFvalue *select)
{
    SWinstance *here = (SWinstance *) inst;

    NG_IGNORE(select);

    switch (param) {
    case SW_IC_ON:
        if (value->iValue)
            here->SWzero_stateGiven = TRUE;
        break;
    case SW_IC_OFF:
        if (value->iValue)
            here->SWzero_stateGiven = FALSE;
        break;
    default:
        return E_BADPARM;
    }

    return OK;
}
