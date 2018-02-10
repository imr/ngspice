/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include "mos6defs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
MOS6delete(GENinstance *gen_inst)
{
    MOS6instance *inst = (MOS6instance *) gen_inst;
    FREE(inst->MOS6sens);
    return OK;
}
