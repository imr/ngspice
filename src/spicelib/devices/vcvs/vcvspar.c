/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

#include "ngspice/ngspice.h"
#include "ngspice/ifsim.h"
#include "vcvsdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


/* ARGSUSED */
int
VCVSparam(int param, IFvalue *value, GENinstance *inst, IFvalue *select)
{
    VCVSinstance *here = (VCVSinstance *)inst;

    NG_IGNORE(select);

    switch(param) {
        case VCVS_GAIN:
            here->VCVScoeff = value->rValue;
            here->VCVScoeffGiven = TRUE;
            break;
        case VCVS_GAIN_SENS:
            here->VCVSsenParmNo = value->iValue;
            break;
        default:
            return(E_BADPARM);
    }
    return(OK);
}
