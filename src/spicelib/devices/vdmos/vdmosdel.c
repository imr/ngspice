/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include "vdmosdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
VDMOSdelete(GENinstance *gen_inst)
{
    VDMOSinstance *inst = (VDMOSinstance *) gen_inst;
    return OK;
}
