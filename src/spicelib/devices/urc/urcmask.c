/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1987 Thomas L. Quarles
**********/
/*
 */

/*
 * This routine gives access to the internal model parameters
 * of Uniform distributed RC lines
 */

#include "ngspice.h"
#include "cktdefs.h"
#include "devdefs.h"
#include "ifsim.h"
#include "urcdefs.h"
#include "sperror.h"
#include "suffix.h"


/* ARGSUSED */
int
URCmAsk(CKTcircuit *ckt, GENmodel *inst, int which, IFvalue *value)
{
    URCmodel *here = (URCmodel *)inst;
    switch(which) {
        case URC_MOD_K:
            value->rValue = here->URCk;
            return (OK);
        case URC_MOD_FMAX:
            value->rValue = here->URCfmax;
            return (OK);
        case URC_MOD_RPERL:
            value->rValue = here->URCrPerL;
            return (OK);
        case URC_MOD_CPERL:
            value->rValue = here->URCcPerL;
            return (OK);
        case URC_MOD_ISPERL:
            value->rValue = here->URCisPerL;
            return (OK);
        case URC_MOD_RSPERL:
            value->rValue = here->URCrsPerL;
            return (OK);
        default:
            return (E_BADPARM);
    }
    /* NOTREACHED */
}
