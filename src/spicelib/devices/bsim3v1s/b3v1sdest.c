/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1995 Min-Chie Jeng and Mansun Chan.
Modified by Paolo Nenzi 2002
File: b3v1sdest.c
**********/

#include "ngspice.h"
#include "bsim3v1sdef.h"
#include "suffix.h"

void
BSIM3v1Sdestroy(GENmodel **inModel)
{
BSIM3v1Smodel **model = (BSIM3v1Smodel**)inModel;
BSIM3v1Sinstance *here;
BSIM3v1Sinstance *prev = NULL;
BSIM3v1Smodel *mod = *model;
BSIM3v1Smodel *oldmod = NULL;

    for (; mod ; mod = mod->BSIM3v1SnextModel)
    {    if(oldmod) FREE(oldmod);
         oldmod = mod;
         prev = (BSIM3v1Sinstance *)NULL;
         for (here = mod->BSIM3v1Sinstances; here; here = here->BSIM3v1SnextInstance)
	 {    if(prev) FREE(prev);
              prev = here;
         }
         if(prev) FREE(prev);
    }
    if(oldmod) FREE(oldmod);
    *model = NULL;
    return;
}



