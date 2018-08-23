/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include "isrcdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"
#include "ngspice/1-f-code.h"


int
ISRCdelete(GENinstance *gen_inst)
{
    ISRCinstance *inst = (ISRCinstance *) gen_inst;

    FREE(inst->ISRCcoeffs);
    trnoise_state_free(inst->ISRCtrnoise_state);
    FREE(inst->ISRCtrrandom_state);

    return OK;
}
