/**** BSIM3v3.2.4, Released by Xuemei Xi 12/14/2001 ****/

/**********
 * Copyright 2001 Regents of the University of California. All rights reserved.
 * File: b3dest.c of BSIM3v3.2.4
 * Author: 1995 Min-Chie Jeng and Mansun Chan. 
 * Author: 1997-1999 Weidong Liu.
 * Author: 2001  Xuemei Xi
 * Modified by Paolo Nenzi 2002
 **********/

#include "ngspice.h"
#include "bsim3def.h"
#include "suffix.h"

void
BSIM3destroy (GENmodel **inModel)
{
BSIM3model **model = (BSIM3model**)inModel;
BSIM3instance *here;
BSIM3instance *prev = NULL;
BSIM3model *mod = *model;
BSIM3model *oldmod = NULL;

    for (; mod ; mod = mod->BSIM3nextModel)
    {    if(oldmod) FREE(oldmod);
         oldmod = mod;
         prev = (BSIM3instance *)NULL;
         for (here = mod->BSIM3instances; here; here = here->BSIM3nextInstance)
	 {    if(prev) FREE(prev);
              prev = here;
         }
         if(prev) FREE(prev);
    }
    if(oldmod) FREE(oldmod);
    *model = NULL;
    return;
}
