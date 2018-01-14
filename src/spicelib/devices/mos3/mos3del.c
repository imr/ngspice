/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include "mos3defs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
MOS3delete(GENinstance *inst)
{
    MOS3instance *here = (MOS3instance *) inst;
    FREE(here->MOS3sens);
    GENinstanceFree(inst);
    return OK;
}
