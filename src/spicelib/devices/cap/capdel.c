/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: September 2003 Paolo Nenzi
**********/

#include "ngspice/ngspice.h"
#include "capdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
CAPdelete(GENinstance *inst)
{
    GENinstanceFree(inst);
    return OK;
}
