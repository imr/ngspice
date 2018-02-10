/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1987 Kanwar Jit Singh
**********/

#include "ngspice/ngspice.h"
#include "asrcdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/inpdefs.h"
#include "ngspice/suffix.h"


int
ASRCdelete(GENinstance *gen_inst)
{
    ASRCinstance *inst = (ASRCinstance *) gen_inst;

    INPfreeTree(inst->ASRCtree);
    FREE(inst->ASRCacValues);
    FREE(inst->ASRCposPtr);
    FREE(inst->ASRCvars);
    return OK;
}
