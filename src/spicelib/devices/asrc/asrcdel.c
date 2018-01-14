/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1987 Kanwar Jit Singh
**********/

#include "ngspice/ngspice.h"
#include "asrcdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"
#include "ngspice/inpdefs.h"


int
ASRCdelete(GENinstance *inst)
{
    ASRCinstance *here = (ASRCinstance *) inst;

    INPfreeTree(here->ASRCtree);
    FREE(here->ASRCacValues);
    FREE(here->ASRCposPtr);
    FREE(here->ASRCvars);

    GENinstanceFree(inst);
    return OK;
}
