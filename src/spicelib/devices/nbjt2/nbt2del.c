/**********
Copyright 1992 Regents of the University of California.  All rights reserved.
Author: 1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
**********/

/*
 * This routine deletes a NBJT2 instance from the circuit and frees the
 * storage it was using.
 */

#include "ngspice/ngspice.h"
#include "nbjt2def.h"
#include "../../../ciderlib/twod/twoddefs.h"
#include "../../../ciderlib/twod/twodext.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
NBJT2delete(GENinstance *inst)
{
    NBJT2instance *here = (NBJT2instance *) inst;
    TWOdestroy(here->NBJT2pDevice);
    GENinstanceFree(inst);
    return OK;
}
