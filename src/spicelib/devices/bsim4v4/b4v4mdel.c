/**** BSIM4.4.0  Released by Xuemei (Jane) Xi 03/04/2004 ****/

/**********
 * Copyright 2004 Regents of the University of California. All rights reserved.
 * File: b4mdel.c of BSIM4.4.0.
 * Author: 2000 Weidong Liu
 * Authors: 2001- Xuemei Xi, Jin He, Kanyu Cao, Mohan Dunga, Mansun Chan, Ali Niknejad, Chenming Hu.
 * Project Director: Prof. Chenming Hu.
 **********/

#include "ngspice/ngspice.h"
#include "bsim4v4def.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int
BSIM4v4mDelete(
GENmodel **inModel,
IFuid modname,
GENmodel *kill)
{
BSIM4v4model **model = (BSIM4v4model**)inModel;
BSIM4v4model *modfast = (BSIM4v4model*)kill;
BSIM4v4instance *here;
BSIM4v4instance *prev = NULL;
BSIM4v4model **oldmod;

    oldmod = model;
    for (; *model ; model = &((*model)->BSIM4v4nextModel)) 
    {    if ((*model)->BSIM4v4modName == modname || 
             (modfast && *model == modfast))
	     goto delgot;
         oldmod = model;
    }
    return(E_NOMOD);

delgot:
    *oldmod = (*model)->BSIM4v4nextModel; /* cut deleted device out of list */
    for (here = (*model)->BSIM4v4instances; here; here = here->BSIM4v4nextInstance)
    {    if(prev) FREE(prev);
         prev = here;
    }
    if(prev) FREE(prev);
    FREE(*model);
    return(OK);
}
