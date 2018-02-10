/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include "mos2defs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
MOS2delete(GENinstance *gen_inst)
{
    MOS2instance *inst = (MOS2instance *) gen_inst;
    FREE(inst->MOS2sens);
    return OK;
}
