/**** BSIM4.4.0  Released by Xuemei (Jane) Xi 03/04/2004 ****/

/**********
 * Copyright 2004 Regents of the University of California. All rights reserved.
 * File: b4dest.c of BSIM4.4.0.
 * Author: 2000 Weidong Liu
 * Authors: 2001- Xuemei Xi, Jin He, Kanyu Cao, Mohan Dunga, Mansun Chan, Ali Niknejad, Chenming Hu.
 * Project Director: Prof. Chenming Hu.
 **********/

#include "ngspice.h"
#include "bsim4v4def.h"
#include "suffix.h"

void
BSIM4V4destroy(inModel)
GENmodel **inModel;
{
BSIM4V4model **model = (BSIM4V4model**)inModel;
BSIM4V4instance *here;
BSIM4V4instance *prev = NULL;
BSIM4V4model *mod = *model;
BSIM4V4model *oldmod = NULL;

    for (; mod ; mod = mod->BSIM4V4nextModel)
    {    if(oldmod) FREE(oldmod);
         oldmod = mod;
         prev = (BSIM4V4instance *)NULL;
         for (here = mod->BSIM4V4instances; here; here = here->BSIM4V4nextInstance)
	 {    if(prev) FREE(prev);
              prev = here;
         }
         if(prev) FREE(prev);
    }
    if(oldmod) FREE(oldmod);
    *model = NULL;
    return;
}
