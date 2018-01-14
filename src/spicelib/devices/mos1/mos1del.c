/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include "mos1defs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
MOS1delete(GENinstance *inst)
{
    MOS1instance *here = (MOS1instance *) inst;
    FREE(here->MOS1sens);
    GENinstanceFree(inst);
    return OK;
}
