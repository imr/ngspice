/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1987 Kanwar Jit Singh
**********/

#include "ngspice/ngspice.h"
#include "asrcdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
ASRCmDelete(GENmodel **inModel, IFuid modname, GENmodel *kill)
{
    ASRCmodel **model = (ASRCmodel **) inModel;
    ASRCmodel *modfast = (ASRCmodel *) kill;
    ASRCinstance *here;
    ASRCinstance *prev = NULL;
    ASRCmodel **oldmod;

    oldmod = model;
    for (; *model; model = &((*model)->ASRCnextModel)) {
        if ((*model)->ASRCmodName == modname || (modfast && *model == modfast))
            goto delgot;
        oldmod = model;
    }

    return E_NOMOD;

 delgot:

    *oldmod = (*model)->ASRCnextModel; /* cut deleted device out of list */

    for (here = (*model)->ASRCinstances; here; here = here->ASRCnextInstance) {
        FREE(here->ASRCacValues);
        if (prev)
            FREE(prev);
        prev = here;
    }

    if (prev)
        FREE(prev);
    FREE(*model);
    return OK;
}
