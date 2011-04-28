/**** BSIM4.5.0 Released by Xuemei (Jane) Xi 07/29/2005 ****/

/**********
 * Copyright 2005 Regents of the University of California. All rights reserved.
 * File: b4dest.c of BSIM4.5.0.
 * Author: 2000 Weidong Liu
 * Authors: 2001- Xuemei Xi, Mohan Dunga, Ali Niknejad, Chenming Hu.
 * Project Director: Prof. Chenming Hu.
 **********/

#include "ngspice.h"
#include "bsim4v5def.h"
#include "suffix.h"

void
BSIM4v5destroy(
GENmodel **inModel)
{
BSIM4v5model **model = (BSIM4v5model**)inModel;
BSIM4v5instance *here;
BSIM4v5instance *prev = NULL;
BSIM4v5model *mod = *model;
BSIM4v5model *oldmod = NULL;

    for (; mod ; mod = mod->BSIM4v5nextModel)
    {    if(oldmod) FREE(oldmod);
         oldmod = mod;
         prev = NULL;
         for (here = mod->BSIM4v5instances; here; here = here->BSIM4v5nextInstance)
	 {    if(prev) FREE(prev);
              prev = here;
         }
         if(prev) FREE(prev);
    }
    if(oldmod) FREE(oldmod);
    *model = NULL;
    return;
}
