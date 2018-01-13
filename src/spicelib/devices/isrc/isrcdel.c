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
    GENinstanceFree(inst);
    return OK;
}
