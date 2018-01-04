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
BSIM4v5mDelete(GENmodel **model, IFuid modname, GENmodel *kill)
{
    GENinstance *here;
    GENinstance *prev = NULL;
    GENmodel **oldmod;

    oldmod = model;
    for (; *model; model = &((*model)->GENnextModel)) {
        if ((*model)->GENmodName == modname || (kill && *model == kill))
            goto delgot;
        oldmod = model;
    }

    return E_NOMOD;

 delgot:
    *oldmod = (*model)->GENnextModel; /* cut deleted device out of list */
    for (here = (*model)->GENinstances; here; here = here->GENnextInstance) {
        if (prev)
            FREE(prev);
        prev = here;
    }
    if (prev)
        FREE(prev);
    FREE(*model);
    return OK;
}
