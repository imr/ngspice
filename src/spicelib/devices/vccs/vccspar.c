/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

#include "ngspice.h"
#include "ifsim.h"
#include "vccsdefs.h"
#include "sperror.h"
#include "suffix.h"


/* ARGSUSED */
int
VCCSparam(int param, IFvalue *value, GENinstance *inst, IFvalue *select)
{
    VCCSinstance *here = (VCCSinstance *)inst;
    switch(param) {
        case VCCS_TRANS:
            here->VCCScoeff = value->rValue;
            here->VCCScoeffGiven = TRUE;
            break;
        case VCCS_TRANS_SENS:
            here->VCCSsenParmNo = value->iValue;
            break;
        default:
            return(E_BADPARM);
    }
    return(OK);
}
