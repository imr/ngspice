/**** BSIM4.2.1, Released by Xuemei Xi 10/05/2001 ****/

/**********
 * Copyright 2001 Regents of the University of California. All rights reserved.
 * File: b4mdel.c of BSIM4.2.1.
 * Author: 2000 Weidong Liu
 * Authors: Xuemei Xi, Kanyu M. Cao, Hui Wan, Mansun Chan, Chenming Hu.
 * Project Director: Prof. Chenming Hu.
 **********/

#include "ngspice.h"
#include <stdio.h>
#include "bsim4v2def.h"
#include "sperror.h"


int
BSIM4v2mDelete(inModel,modname,kill)
GENmodel **inModel;
IFuid modname;
GENmodel *kill;
{
BSIM4v2model **model = (BSIM4v2model**)inModel;
BSIM4v2model *modfast = (BSIM4v2model*)kill;
BSIM4v2instance *here;
BSIM4v2instance *prev = NULL;
BSIM4v2model **oldmod;

    oldmod = model;
    for (; *model ; model = &((*model)->BSIM4v2nextModel)) 
    {    if ((*model)->BSIM4v2modName == modname || 
             (modfast && *model == modfast))
	     goto delgot;
         oldmod = model;
    }
    return(E_NOMOD);

delgot:
    *oldmod = (*model)->BSIM4v2nextModel; /* cut deleted device out of list */
    for (here = (*model)->BSIM4v2instances; here; here = here->BSIM4v2nextInstance)
    {    if(prev) FREE(prev);
         prev = here;
    }
    if(prev) FREE(prev);
    FREE(*model);
    return(OK);
}
