/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include "isrcdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
ISRCdelete(GENinstance *inst)
{
    ISRCinstance *here = (ISRCinstance *) inst;
    FREE(here->ISRCcoeffs);
    GENinstanceFree(inst);
    return OK;
}
