/**********
Author: Florian Ballenegger 2020
Adapted from VCVS device code.
**********/
/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1987 Thomas L. Quarles
**********/

/*
 * This routine gives access to the internal device parameters
 * of Voltage Controlled Voltage Source
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/devdefs.h"
#include "ngspice/ifsim.h"
#include "nuladefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

/* ARGSUSED */
int
NULAask(CKTcircuit *ckt, GENinstance *inst, int which, IFvalue *value, IFvalue *select)
{
    NULAinstance *here = (NULAinstance *)inst;
    switch(which) {
        case NULA_CONT_P_NODE:
            value->iValue = here->NULAcontPosNode;
            return (OK);
        case NULA_CONT_N_NODE:
            value->iValue = here->NULAcontNegNode;
            return (OK);
        default:
            return (E_BADPARM);
    }
    /* NOTREACHED */
}
