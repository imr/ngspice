/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1995 Min-Chie Jeng and Mansun Chan.
File: b3v0dest.c
**********/

#include "ngspice.h"
#include "bsim3v0def.h"
#include "suffix.h"

void
BSIM3v0destroy(GENmodel **inModel)
{
BSIM3v0model **model = (BSIM3v0model**)inModel;
BSIM3v0instance *here;
BSIM3v0instance *prev = NULL;
BSIM3v0model *mod = *model;
BSIM3v0model *oldmod = NULL;

    for (; mod ; mod = mod->BSIM3v0nextModel)
    {    if(oldmod) FREE(oldmod);
         oldmod = mod;
         prev = (BSIM3v0instance *)NULL;
         for (here = mod->BSIM3v0instances; here; here = here->BSIM3v0nextInstance)
	 {    if(prev) FREE(prev);
              prev = here;
         }
         if(prev) FREE(prev);
    }
    if(oldmod) FREE(oldmod);
    *model = NULL;
    return;
}



