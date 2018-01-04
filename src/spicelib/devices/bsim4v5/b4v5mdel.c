/**** BSIM4.5.0 Released by Xuemei (Jane) Xi 07/29/2005 ****/

/**********
 * Copyright 2005 Regents of the University of California. All rights reserved.
 * File: b4mdel.c of BSIM4.5.0.
 * Author: 2000 Weidong Liu
 * Authors: 2001- Xuemei Xi, Mohan Dunga, Ali Niknejad, Chenming Hu.
 * Project Director: Prof. Chenming Hu.
 **********/

#include "ngspice/ngspice.h"
#include "bsim4v5def.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
BSIM4v5mDelete(
               GENmodel **inModel,
               IFuid modname,
               GENmodel *kill)
{
    BSIM4v5model **model = (BSIM4v5model **) inModel;
    BSIM4v5model *modfast = (BSIM4v5model *) kill;
    BSIM4v5instance *here;
    BSIM4v5instance *prev = NULL;
    BSIM4v5model **oldmod;

    oldmod = model;
    for (; *model; model = &((*model)->BSIM4v5nextModel)) {
        if ((*model)->BSIM4v5modName == modname ||
            (modfast && *model == modfast))
            goto delgot;
        oldmod = model;
    }

    return E_NOMOD;

 delgot:
    *oldmod = (*model)->BSIM4v5nextModel; /* cut deleted device out of list */
    for (here = (*model)->BSIM4v5instances; here; here = here->BSIM4v5nextInstance) {
        if (prev) FREE(prev);
        prev = here;
    }
    if (prev) FREE(prev);
    FREE(*model);
    return OK;
}
