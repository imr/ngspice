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
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
NBJTdelete(GENinstance *gen_inst)
{
    NG_IGNORE(gen_inst);
    return OK;
}
