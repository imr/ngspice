/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1987 Thomas L. Quarles
**********/
/*
 */


/*
 * This routine gives access to the internal device parameters
 * of Uniform distributed RC lines
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/devdefs.h"
#include "ngspice/ifsim.h"
#include "urcdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


/* ARGSUSED */
int
URCask(CKTcircuit *ckt, GENinstance *inst, int which, IFvalue *value, IFvalue *select)
{
    URCinstance *here = (URCinstance *)inst;

    NG_IGNORE(select);
    NG_IGNORE(ckt);

    switch(which) {
        case URC_POS_NODE:
            value->iValue = here->URCposNode;
            return (OK);
        case URC_NEG_NODE:
            value->iValue = here->URCnegNode;
            return (OK);
        case URC_GND_NODE:
            value->iValue = here->URCgndNode;
            return (OK);
        case URC_LEN:
            value->rValue = here->URClength;
            return (OK);
        case URC_LUMPS:
            value->iValue = here->URClumps;
            return (OK);
        default:
            return (E_BADPARM);
    }
    /* NOTREACHED */
}
