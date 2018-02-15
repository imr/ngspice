/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 S. Hwang
**********/

#include "ngspice/ngspice.h"
#include "mesdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
MESmDelete(GENmodel *gen_model)
{
    NG_IGNORE(gen_model);
    return OK;
}
