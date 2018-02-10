/**********
Copyright 1992 Regents of the University of California.  All rights reserved.
Author: 1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
**********/

#include "ngspice/ngspice.h"
#include "numddefs.h"
#include "../../../ciderlib/oned/onedext.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
NUMDdelete(GENinstance *gen_inst)
{
    NUMDinstance *inst = (NUMDinstance *) gen_inst;

    ONEdestroy(inst->NUMDpDevice);

    return OK;
}
