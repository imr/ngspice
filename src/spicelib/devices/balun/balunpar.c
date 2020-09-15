/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

#include "ngspice/ngspice.h"
#include "ngspice/ifsim.h"
#include "balundefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


/* ARGSUSED */
int
BALUNparam(int param, IFvalue *value, GENinstance *inst, IFvalue *select)
{
    BALUNinstance *here = (BALUNinstance *)inst;

    NG_IGNORE(select);

    switch(param) {
        default:
            return(E_BADPARM);
    }
    return(OK);
}
