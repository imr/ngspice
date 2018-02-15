/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Gordon Jacobs
**********/

#include "ngspice/ngspice.h"
#include "cswdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
CSWmDelete(GENmodel *gen_model)
{
    NG_IGNORE(gen_model);
    return OK;
}
