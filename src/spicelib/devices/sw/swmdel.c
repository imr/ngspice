/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Gordon Jacobs
**********/

#include "ngspice/ngspice.h"
#include "swdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
SWmDelete(GENmodel *gen_model)
{
    NG_IGNORE(gen_model);
    return OK;
}
