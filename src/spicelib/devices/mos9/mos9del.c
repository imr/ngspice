/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: Alan Gillespie
**********/

#include "ngspice/ngspice.h"
#include "mos9defs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
MOS9delete(GENinstance *gen_inst)
{
    MOS9instance *inst = (MOS9instance *) gen_inst;
    FREE(inst->MOS9sens);
    return OK;
}
