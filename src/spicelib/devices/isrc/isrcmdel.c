/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include "isrcdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
ISRCmDelete(GENmodel *gen_model)
{
    NG_IGNORE(gen_model);
    return OK;
}
