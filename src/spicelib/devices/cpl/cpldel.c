/**********
Copyright 1992 Regents of the University of California.  All rights
reserved.
Author: 1992 Charles Hough
**********/

#include "ngspice/ngspice.h"
#include "cpldefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
CPLdelete(GENinstance *inst)
{
    GENinstanceFree(inst);
    return OK;
}
