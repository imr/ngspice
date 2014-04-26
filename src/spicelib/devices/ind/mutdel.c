/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include "inddefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
MUTdelete(GENinstance *gen_inst)
{
    NG_IGNORE(gen_inst);
    return OK;
}
