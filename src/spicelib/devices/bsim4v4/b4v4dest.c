/**** BSIM4.4.0  Released by Xuemei (Jane) Xi 03/04/2004 ****/

/**********
 * Copyright 2004 Regents of the University of California. All rights reserved.
 * File: b4dest.c of BSIM4.4.0.
 * Author: 2000 Weidong Liu
 * Authors: 2001- Xuemei Xi, Jin He, Kanyu Cao, Mohan Dunga, Mansun Chan, Ali Niknejad, Chenming Hu.
 * Project Director: Prof. Chenming Hu.
 **********/

#include "ngspice/ngspice.h"
#include "bsim4v4def.h"
#include "ngspice/suffix.h"

void
BSIM4v4destroy(
GENmodel **inModel)
{
BSIM4v4model **model = (BSIM4v4model**)inModel;
BSIM4v4instance *here;
BSIM4v4instance *prev = NULL;
BSIM4v4model *mod = *model;
BSIM4v4model *oldmod = NULL;

#ifdef USE_OMP
    /* free just once for all models */
    FREE(mod->BSIM4v4InstanceArray);
#endif

    for (; mod ; mod = mod->BSIM4v4nextModel)
    {    if(oldmod) FREE(oldmod);
         oldmod = mod;
         prev = NULL;
         for (here = mod->BSIM4v4instances; here; here = here->BSIM4v4nextInstance)
	 {    if(prev) FREE(prev);
              prev = here;
         }
         if(prev) FREE(prev);
    }
    if(oldmod) FREE(oldmod);
    *model = NULL;
    return;
}
