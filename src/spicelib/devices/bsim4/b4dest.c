/**** BSIM4.0.0, Released by Weidong Liu 3/24/2000 ****/

/**********
 * Copyright 2000 Regents of the University of California. All rights reserved.
 * File: b4dest.c of BSIM4.0.0.
 * Authors: Weidong Liu, Kanyu M. Cao, Xiaodong Jin, Chenming Hu.
 * Project Director: Prof. Chenming Hu.
 **********/

#include "ngspice.h"
#include <stdio.h>
#include "bsim4def.h"

void
BSIM4destroy(inModel)
GENmodel **inModel;
{
BSIM4model **model = (BSIM4model**)inModel;
BSIM4instance *here;
BSIM4instance *prev = NULL;
BSIM4model *mod = *model;
BSIM4model *oldmod = NULL;

    for (; mod ; mod = mod->BSIM4nextModel)
    {    if(oldmod) FREE(oldmod);
         oldmod = mod;
         prev = (BSIM4instance *)NULL;
         for (here = mod->BSIM4instances; here; here = here->BSIM4nextInstance)
	 {    if(prev) FREE(prev);
              prev = here;
         }
         if(prev) FREE(prev);
    }
    if(oldmod) FREE(oldmod);
    *model = NULL;
    return;
}
