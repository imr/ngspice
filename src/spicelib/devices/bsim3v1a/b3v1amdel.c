/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1995 Min-Chie Jeng and Mansun Chan.
Modified by Paolo Nenzi 2002
File: b3v1amdel.c
**********/

#include "ngspice.h"
#include "bsim3v1adef.h"
#include "sperror.h"
#include "suffix.h"

int
BSIM3v1AmDelete(GENmodel **inModel, IFuid modname, GENmodel *kill)
{
BSIM3v1Amodel **model = (BSIM3v1Amodel**)inModel;
BSIM3v1Amodel *modfast = (BSIM3v1Amodel*)kill;
BSIM3v1Ainstance *here;
BSIM3v1Ainstance *prev = NULL;
BSIM3v1Amodel **oldmod;

    oldmod = model;
    for (; *model ; model = &((*model)->BSIM3v1AnextModel)) 
    {    if ((*model)->BSIM3v1AmodName == modname || 
             (modfast && *model == modfast))
	     goto delgot;
         oldmod = model;
    }
    return(E_NOMOD);

delgot:
    *oldmod = (*model)->BSIM3v1AnextModel; /* cut deleted device out of list */
    for (here = (*model)->BSIM3v1Ainstances; here; here = here->BSIM3v1AnextInstance)
    {    if(prev) FREE(prev);
         prev = here;
    }
    if(prev) FREE(prev);
    FREE(*model);
    return(OK);
}



