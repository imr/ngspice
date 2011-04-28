/**** BSIM4.2.1, Released by Xuemei Xi 10/05/2001 ****/

/**********
 * Copyright 2001 Regents of the University of California. All rights reserved.
 * File: b4dest.c of BSIM4.2.1.
 * Author: 2000 Weidong Liu
 * Authors: Xuemei Xi, Kanyu M. Cao, Hui Wan, Mansun Chan, Chenming Hu.
 * Project Director: Prof. Chenming Hu.
 **********/

#include "ngspice.h"
#include <stdio.h>
#include "bsim4v2def.h"

void
BSIM4v2destroy(
GENmodel **inModel)
{
BSIM4v2model **model = (BSIM4v2model**)inModel;
BSIM4v2instance *here;
BSIM4v2instance *prev = NULL;
BSIM4v2model *mod = *model;
BSIM4v2model *oldmod = NULL;

    for (; mod ; mod = mod->BSIM4v2nextModel)
    {    if(oldmod) FREE(oldmod);
         oldmod = mod;
         prev = NULL;
         for (here = mod->BSIM4v2instances; here; here = here->BSIM4v2nextInstance)
	 {    if(prev) FREE(prev);
              prev = here;
         }
         if(prev) FREE(prev);
    }
    if(oldmod) FREE(oldmod);
    *model = NULL;
    return;
}
