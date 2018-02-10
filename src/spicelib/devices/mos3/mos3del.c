/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include "mos3defs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
MOS3delete(GENinstance *gen_inst)
{
    MOS3instance *inst = (MOS3instance *) gen_inst;
    FREE(inst->MOS3sens);
    return OK;
}
