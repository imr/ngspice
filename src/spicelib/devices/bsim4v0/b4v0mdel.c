/**** BSIM4v0.0.0, Released by Weidong Liu 3/24/2000 ****/

/**********
 * Copyright 2000 Regents of the University of California. All rights reserved.
 * File: b4mdel.c of BSIM4v0.0.0.
 * Authors: Weidong Liu, Kanyu M. Cao, Xiaodong Jin, Chenming Hu.
 * Project Director: Prof. Chenming Hu.
 **********/

#include "ngspice/ngspice.h"
#include "bsim4v0def.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int
BSIM4v0mDelete(inModel,modname,kill)
GENmodel **inModel;
IFuid modname;
GENmodel *kill;
{
BSIM4v0model **model = (BSIM4v0model**)inModel;
BSIM4v0model *modfast = (BSIM4v0model*)kill;
BSIM4v0instance *here;
BSIM4v0instance *prev = NULL;
BSIM4v0model **oldmod;

    oldmod = model;
    for (; *model ; model = &((*model)->BSIM4v0nextModel)) 
    {    if ((*model)->BSIM4v0modName == modname || 
             (modfast && *model == modfast))
	     goto delgot;
         oldmod = model;
    }
    return(E_NOMOD);

delgot:
    *oldmod = (*model)->BSIM4v0nextModel; /* cut deleted device out of list */
    for (here = (*model)->BSIM4v0instances; here; here = here->BSIM4v0nextInstance)
    {    if(prev) FREE(prev);
         prev = here;
    }
    if(prev) FREE(prev);
    FREE(*model);
    return(OK);
}
