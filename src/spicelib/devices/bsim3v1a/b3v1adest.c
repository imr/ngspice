/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1995 Min-Chie Jeng and Mansun Chan.
Modified by Paolo Nenzi 2002
File: b3v1adest.c
**********/

#include "ngspice.h"
#include "bsim3v1adef.h"
#include "suffix.h"

void
BSIM3v1Adestroy(GENmodel **inModel)
{
BSIM3v1Amodel **model = (BSIM3v1Amodel**)inModel;
BSIM3v1Ainstance *here;
BSIM3v1Ainstance *prev = NULL;
BSIM3v1Amodel *mod = *model;
BSIM3v1Amodel *oldmod = NULL;

    for (; mod ; mod = mod->BSIM3v1AnextModel)
    {    if(oldmod) FREE(oldmod);
         oldmod = mod;
         prev = (BSIM3v1Ainstance *)NULL;
         for (here = mod->BSIM3v1Ainstances; here; here = here->BSIM3v1AnextInstance)
	 {    if(prev) FREE(prev);
              prev = here;
         }
         if(prev) FREE(prev);
    }
    if(oldmod) FREE(oldmod);
    *model = NULL;
    return;
}



