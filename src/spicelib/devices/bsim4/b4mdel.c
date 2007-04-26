/**** BSIM4.6.0 Released by Mohan Dunga 12/13/2006 ****/

/**********
 * Copyright 2006 Regents of the University of California. All rights reserved.
 * File: b4mdel.c of BSIM4.6.0.
 * Author: 2000 Weidong Liu
 * Authors: 2001- Xuemei Xi, Mohan Dunga, Ali Niknejad, Chenming Hu.
 * Authors: 2006- Mohan Dunga, Ali Niknejad, Chenming Hu
 * Project Director: Prof. Chenming Hu.
 **********/

#include "ngspice.h"
#include "bsim4def.h"
#include "sperror.h"
#include "suffix.h"

int
BSIM4mDelete(inModel,modname,kill)
GENmodel **inModel;
IFuid modname;
GENmodel *kill;
{
BSIM4model **model = (BSIM4model**)inModel;
BSIM4model *modfast = (BSIM4model*)kill;
BSIM4instance *here;
BSIM4instance *prev = NULL;
BSIM4model **oldmod;

    oldmod = model;
    for (; *model ; model = &((*model)->BSIM4nextModel)) 
    {    if ((*model)->BSIM4modName == modname || 
             (modfast && *model == modfast))
	     goto delgot;
         oldmod = model;
    }
    return(E_NOMOD);

delgot:
    *oldmod = (*model)->BSIM4nextModel; /* cut deleted device out of list */
    for (here = (*model)->BSIM4instances; here; here = here->BSIM4nextInstance)
    {    if(prev) FREE(prev);
         prev = here;
    }
    if(prev) FREE(prev);
    FREE(*model);
    return(OK);
}
