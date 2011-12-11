/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

#include "ngspice/ngspice.h"
#include "ngspice/ifsim.h"
#include "ccvsdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


/* ARGSUSED */
int
CCVSparam(int param, IFvalue *value, GENinstance *inst, IFvalue *select)
{
    CCVSinstance *here = (CCVSinstance*)inst;

    NG_IGNORE(select);

    switch(param) {
        case CCVS_TRANS:
            here->CCVScoeff = value->rValue;
            here->CCVScoeffGiven = TRUE;
            break;
        case CCVS_CONTROL:
            here->CCVScontName = value->uValue;
            break;
        case CCVS_TRANS_SENS:
            here->CCVSsenParmNo = value->iValue;
            break;
        default:
            return(E_BADPARM);
    }
    return(OK);
}
