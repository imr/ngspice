/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1987 Thomas L. Quarles
**********/
/*
 */

#include "ngspice.h"
#include "ifsim.h"
#include "urcdefs.h"
#include "sperror.h"
#include "suffix.h"


/* ARGSUSED */
int
URCparam(int param, IFvalue *value, GENinstance *inst, IFvalue *select)
{
    URCinstance *here = (URCinstance *)inst;
    switch(param) {
        case URC_LEN:
            here->URClength = value->rValue;
            here->URClenGiven = TRUE;
            break;
        case URC_LUMPS:
            here->URClumps = value->rValue;
            here->URClumpsGiven = TRUE;
            break;
        default:
            return(E_BADPARM);
    }
    return(OK);
}
