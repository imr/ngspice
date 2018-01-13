/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Gordon Jacobs
**********/

#include "ngspice/ngspice.h"
#include "cswdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
CSWdelete(GENinstance *inst)
{
    GENinstanceFree(inst);
    return OK;
}
