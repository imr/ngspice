/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1987 Kanwar Jit Singh
**********/

#include "ngspice/ngspice.h"
#include "asrcdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
ASRCdelete(GENinstance *inst)
{
    GENinstanceFree(inst);
    return OK;
}
