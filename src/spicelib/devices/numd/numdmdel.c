/**********
Copyright 1992 Regents of the University of California.  All rights reserved.
Author: 1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
**********/

#include "ngspice/ngspice.h"
#include "numddefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
NUMDmDelete(GENmodel **inModel, IFuid modname, GENmodel *kill)
{
    NUMDmodel **model = (NUMDmodel **) inModel;
    NUMDmodel *modfast = (NUMDmodel *) kill;
    NUMDinstance *here;
    NUMDinstance *prev = NULL;
    NUMDmodel **oldmod;

    oldmod = model;
    for (; *model; model = &((*model)->NUMDnextModel)) {
        if ((*model)->NUMDmodName == modname || (modfast && *model == modfast))
            goto delgot;
        oldmod = model;
    }

    return E_NOMOD;

 delgot:
    *oldmod = (*model)->NUMDnextModel;    /* cut deleted device out of list */
    for (here = (*model)->NUMDinstances; here; here = here->NUMDnextInstance) {
        if (prev)
            FREE(prev);
        prev = here;
    }
    if (prev)
        FREE(prev);
    FREE(*model);
    return OK;
}
