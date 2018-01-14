/**********
Copyright 1992 Regents of the University of California.  All rights reserved.
Author: 1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
**********/

/*
 * This routine deletes a NBJT instance from the circuit and frees the
 * storage it was using.
 */

#include "ngspice/ngspice.h"
#include "nbjtdefs.h"
#include "../../../ciderlib/oned/onedext.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
NBJTdelete(GENinstance *inst)
{
    NBJTinstance *here = (NBJTinstance *) inst;
    ONEdestroy(here->NBJTpDevice);
    GENinstanceFree(inst);
    return OK;
}
