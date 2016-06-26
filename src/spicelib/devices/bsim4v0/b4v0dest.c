/**** BSIM4.0.0, Released by Weidong Liu 3/24/2000 ****/

/**********
 * Copyright 2000 Regents of the University of California. All rights reserved.
 * File: b4dest.c of BSIM4.0.0.
 * Authors: Weidong Liu, Kanyu M. Cao, Xiaodong Jin, Chenming Hu.
 * Project Director: Prof. Chenming Hu.
 **********/

#include "ngspice/ngspice.h"
#include "bsim4v0def.h"
#include "ngspice/suffix.h"

void
BSIM4v0destroy(inModel)
GENmodel **inModel;
{
BSIM4v0model **model = (BSIM4v0model**)inModel;
BSIM4v0instance *here;
BSIM4v0instance *prev = NULL;
BSIM4v0model *mod = *model;
BSIM4v0model *oldmod = NULL;

    for (; mod ; mod = mod->BSIM4v0nextModel)
    {    if(oldmod) FREE(oldmod);
         oldmod = mod;
         prev = NULL;
         for (here = mod->BSIM4v0instances; here; here = here->BSIM4v0nextInstance)
	 {    if(prev) FREE(prev);
              prev = here;
         }
         if(prev) FREE(prev);
    }
    if(oldmod) FREE(oldmod);
    *model = NULL;
    return;
}
