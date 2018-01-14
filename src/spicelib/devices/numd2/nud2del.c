/**********
Copyright 1992 Regents of the University of California.  All rights reserved.
Author: 1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
**********/

#include "ngspice/ngspice.h"
#include "numd2def.h"
#include "../../../ciderlib/twod/twoddefs.h"
#include "../../../ciderlib/twod/twodext.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
NUMD2delete(GENinstance *inst)
{
    NUMD2instance *here = (NUMD2instance *) inst;
    TWOdestroy(here->NUMD2pDevice);
    GENinstanceFree(inst);
    return OK;
}
