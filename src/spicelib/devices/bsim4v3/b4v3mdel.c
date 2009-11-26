/**** BSIM4.3.0 Released by Xuemei(Jane) Xi 05/09/2003 ****/

/**********
 * Copyright 2003 Regents of the University of California. All rights reserved.
 * File: b4v3mdel.c of BSIM4.3.0.
 * Author: 2000 Weidong Liu
 * Authors: 2001- Xuemei Xi, Jin He, Kanyu Cao, Mohan Dunga, Mansun Chan, Ali Niknejad, Chenming Hu.
 * Project Director: Prof. Chenming Hu.
 **********/

#include "ngspice.h"
#include <stdio.h>
#include "bsim4v3def.h"
#include "sperror.h"

int
BSIM4v3mDelete(
GENmodel **inModel,
IFuid modname,
GENmodel *kill)
{
BSIM4v3model **model = (BSIM4v3model**)inModel;
BSIM4v3model *modfast = (BSIM4v3model*)kill;
BSIM4v3instance *here;
BSIM4v3instance *prev = NULL;
BSIM4v3model **oldmod;

    oldmod = model;
    for (; *model ; model = &((*model)->BSIM4v3nextModel)) 
    {    if ((*model)->BSIM4v3modName == modname || 
             (modfast && *model == modfast))
	     goto delgot;
         oldmod = model;
    }
    return(E_NOMOD);

delgot:
    *oldmod = (*model)->BSIM4v3nextModel; /* cut deleted device out of list */
    for (here = (*model)->BSIM4v3instances; here; here = here->BSIM4v3nextInstance)
    {    if(prev) FREE(prev);
         prev = here;
    }
    if(prev) FREE(prev);
    FREE(*model);
    return(OK);
}
