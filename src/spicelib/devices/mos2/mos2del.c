/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include "mos2defs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
MOS2delete(GENinstance *inst)
{
    MOS2instance *here = (MOS2instance *) inst;
    FREE(here->MOS2sens);
    GENinstanceFree(inst);
    return OK;
}
