/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: Apr 2000 - Paolo Nenzi
**********/

#include "ngspice/ngspice.h"
#include "resdefs.h"
#include "ngspice/sperror.h"


int
RESmDelete(GENmodel *gen_model)
{
    NG_IGNORE(gen_model);
    return OK;
}
