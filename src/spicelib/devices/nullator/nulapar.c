/**********
Author: Florian Ballenegger 2020
Adapted from VCVS device code.
**********/
/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

#include "ngspice/ngspice.h"
#include "ngspice/ifsim.h"
#include "nuladefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


/* ARGSUSED */
int
NULAparam(int param, IFvalue *value, GENinstance *inst, IFvalue *select)
{
    NULAinstance *here = (NULAinstance *)inst;

    NG_IGNORE(select);

    switch(param) {
	case NULA_OFFSET:
            here->NULAoffset = value->rValue;
            here->NULAoffsetGiven = TRUE;
            break;
        default:
            return(E_BADPARM);
    }
    return(OK);
}
