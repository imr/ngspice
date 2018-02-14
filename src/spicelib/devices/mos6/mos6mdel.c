/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include "mos6defs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
MOS6mDelete(GENmodel **inModel, IFuid modname, GENmodel *kill)
{
    MOS6model **model = (MOS6model **) inModel;
    MOS6model *modfast = (MOS6model *) kill;
    MOS6instance *here;
    MOS6instance *prev = NULL;
    MOS6model **oldmod;

    oldmod = model;
    for (; *model; model = &((*model)->MOS6nextModel)) {
        if ((*model)->MOS6modName == modname ||
            (modfast && *model == modfast)) goto delgot;
        oldmod = model;
    }

    return(E_NOMOD);

 delgot:
    *oldmod = (*model)->MOS6nextModel; /* cut deleted device out of list */
    for (here = (*model)->MOS6instances; here; here = here->MOS6nextInstance) {
        if (prev) FREE(prev);
        prev = here;
    }
    if (prev) FREE(prev);
    FREE(*model);
    return(OK);
}
