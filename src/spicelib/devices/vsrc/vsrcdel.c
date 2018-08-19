/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include "vsrcdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
VSRCdelete(GENinstance *gen_inst)
{
    VSRCinstance *inst = (VSRCinstance *) gen_inst;

    FREE(inst->VSRCcoeffs);

    FREE(inst->VSRCtrnoise_state);
    FREE(inst->VSRCtrrandom_state);

    return OK;
}
