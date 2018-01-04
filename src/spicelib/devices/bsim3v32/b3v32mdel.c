/**** BSIM3v3.2.4, Released by Xuemei Xi 12/21/2001 ****/

/**********
 * Copyright 2001 Regents of the University of California. All rights reserved.
 * File: b3mdel.c of BSIM3v3.2.4
 * Author: 1995 Min-Chie Jeng and Mansun Chan.
 * Author: 1997-1999 Weidong Liu.
 * Author: 2001  Xuemei Xi
 * Modified by Paolo Nenzi 2002
 **********/

#include "ngspice/ngspice.h"
#include "bsim3v32def.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
BSIM3v32mDelete(GENmodel **inModel, IFuid modname, GENmodel *kill)
{
    BSIM3v32model **model = (BSIM3v32model **) inModel;
    BSIM3v32model *modfast = (BSIM3v32model *) kill;
    BSIM3v32instance *here;
    BSIM3v32instance *prev = NULL;
    BSIM3v32model **oldmod;

    oldmod = model;
    for (; *model; model = &((*model)->BSIM3v32nextModel)) {
        if ((*model)->BSIM3v32modName == modname ||
            (modfast && *model == modfast))
            goto delgot;
        oldmod = model;
    }

    return E_NOMOD;

 delgot:
    *oldmod = (*model)->BSIM3v32nextModel; /* cut deleted device out of list */
    for (here = (*model)->BSIM3v32instances; here; here = here->BSIM3v32nextInstance) {
        if (prev) FREE(prev);
        prev = here;
    }
    if (prev) FREE(prev);
    FREE(*model);
    return OK;
}
