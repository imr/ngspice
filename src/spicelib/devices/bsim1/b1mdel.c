/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Hong J. Park, Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include "bsim1def.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
B1mDelete(GENmodel *gen_model)
{
    NG_IGNORE(gen_model);
    return OK;
}
