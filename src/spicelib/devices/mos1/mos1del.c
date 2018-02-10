/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include "mos1defs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
MOS1delete(GENinstance *gen_inst)
{
    MOS1instance *inst = (MOS1instance *) gen_inst;
    FREE(inst->MOS1sens);
    return OK;
}
