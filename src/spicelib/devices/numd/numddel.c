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
NUMDdelete(GENinstance *inst)
{
    NUMDinstance *here = (NUMDinstance *) inst;
    ONEdestroy(here->NUMDpDevice);
    GENinstanceFree(inst);
    return OK;
}
