/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

#include "ngspice/ngspice.h"
#include "ngspice/ifsim.h"
#include "vccsdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


/* ARGSUSED */
int
VCCSparam(int param, IFvalue *value, GENinstance *inst, IFvalue *select)
{
    VCCSinstance *here = (VCCSinstance *)inst;

    NG_IGNORE(select);

    switch(param) {
        case VCCS_TRANS:
            here->VCCScoeff = value->rValue;
            here->VCCScoeffGiven = TRUE;
            if (here->VCCSmGiven)
                here->VCCScoeff *= here->VCCSmValue;
            break;
        case VCCS_M:
            here->VCCSmValue = value->rValue;
            here->VCCSmGiven = TRUE;
            break;
        case VCCS_TRANS_SENS:
            here->VCCSsenParmNo = value->iValue;
            break;
        default:
            return(E_BADPARM);
    }
    return(OK);
}
