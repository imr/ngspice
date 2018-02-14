/**** BSIM4.6.2 Released by Wenwei Yang 07/31/2008 ****/

/**********
 * Copyright 2006 Regents of the University of California. All rights reserved.
 * File: b4mdel.c of BSIM4.6.2.
 * Author: 2000 Weidong Liu
 * Authors: 2001- Xuemei Xi, Mohan Dunga, Ali Niknejad, Chenming Hu.
 * Authors: 2006- Mohan Dunga, Ali Niknejad, Chenming Hu
 * Authors: 2007- Mohan Dunga, Wenwei Yang, Ali Niknejad, Chenming Hu
 * Project Director: Prof. Chenming Hu.
 **********/

#include "ngspice/ngspice.h"
#include "bsim4v6def.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
BSIM4v6mDelete(
               GENmodel **inModel,
               IFuid modname,
               GENmodel *kill)
{
    BSIM4v6model **model = (BSIM4v6model **) inModel;
    BSIM4v6model *modfast = (BSIM4v6model *) kill;
    BSIM4v6instance *here;
    BSIM4v6instance *prev = NULL;
    BSIM4v6model **oldmod;

    oldmod = model;
    for (; *model; model = &((*model)->BSIM4v6nextModel))
    {   if ((*model)->BSIM4v6modName == modname ||
            (modfast && *model == modfast))
            goto delgot;
        oldmod = model;
    }

    return(E_NOMOD);

 delgot:
    *oldmod = (*model)->BSIM4v6nextModel; /* cut deleted device out of list */
    for (here = (*model)->BSIM4v6instances; here; here = here->BSIM4v6nextInstance)
    {   if (prev) FREE(prev);
        prev = here;
    }
    if (prev) FREE(prev);
    FREE(*model);
    return(OK);
}
