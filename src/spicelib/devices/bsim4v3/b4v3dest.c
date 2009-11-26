/**** BSIM4.3.0 Released by Xuemei(Jane) Xi 05/09/2003  ****/

/**********
 * Copyright 2003 Regents of the University of California. All rights reserved.
 * File: b4v3dest.c of BSIM4.3.0.
 * Author: 2000 Weidong Liu
 * Authors: 2001- Xuemei Xi, Jin He, Kanyu Cao, Mohan Dunga, Mansun Chan, Ali Niknejad, Chenming Hu.
 * Project Director: Prof. Chenming Hu.
 **********/

#include "ngspice.h"
#include <stdio.h>
#include "bsim4v3def.h"

void
BSIM4v3destroy(
GENmodel **inModel)
{
BSIM4v3model **model = (BSIM4v3model**)inModel;
BSIM4v3instance *here;
BSIM4v3instance *prev = NULL;
BSIM4v3model *mod = *model;
BSIM4v3model *oldmod = NULL;

    for (; mod ; mod = mod->BSIM4v3nextModel)
    {    if(oldmod) FREE(oldmod);
         oldmod = mod;
         prev = (BSIM4v3instance *)NULL;
         for (here = mod->BSIM4v3instances; here; here = here->BSIM4v3nextInstance)
	 {    if(prev) FREE(prev);
              prev = here;
         }
         if(prev) FREE(prev);
    }
    if(oldmod) FREE(oldmod);
    *model = NULL;
    return;
}
