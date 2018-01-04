/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include "vsrcdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
VSRCmDelete(GENmodel **inModel, IFuid modname, GENmodel *fast)
{
    VSRCmodel **model = (VSRCmodel **) inModel;
    VSRCmodel *modfast = (VSRCmodel *) fast;
    VSRCinstance *here;
    VSRCinstance *prev = NULL;
    VSRCmodel **oldmod;

    oldmod = model;
    for (; *model; model = &((*model)->VSRCnextModel)) {
        if ((*model)->VSRCmodName == modname ||
            (modfast && *model == modfast)) goto delgot;
        oldmod = model;
    }

    return E_NOMOD;

 delgot:
    *oldmod = (*model)->VSRCnextModel; /* cut deleted device out of list */
    for (here = (*model)->VSRCinstances; here; here = here->VSRCnextInstance) {
        if (prev) FREE(prev);
        prev = here;
    }
    if (prev) FREE(prev);
    FREE(*model);
    return OK;
}
