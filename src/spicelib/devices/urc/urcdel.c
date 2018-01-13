/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1987 Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include "urcdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
URCdelete(GENinstance *inst)
{
    GENinstanceFree(inst);
    return OK;
}
