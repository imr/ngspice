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
NUMD2delete(GENinstance *gen_inst)
{
    NUMD2instance *inst = (NUMD2instance *) gen_inst;

    TWOdestroy(inst->NUMD2pDevice);

    return OK;
}
