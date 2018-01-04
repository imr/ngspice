/**********
Copyright 1992 Regents of the University of California.  All rights reserved.
Author: 1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
**********/

#include "ngspice/ngspice.h"
#include "ndevdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
NDEVmDelete(GENmodel **model, IFuid modname, GENmodel *kill)
{
    NG_IGNORE(model);
    NG_IGNORE(modname);
    NG_IGNORE(kill);

    return OK;
}
