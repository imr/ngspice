/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1995 Min-Chie Jeng and Mansun Chan.
Modified by Paolo Nenzi 2002
File: b3v1mdel.c
**********/

#include "ngspice.h"
#include "bsim3v1sdef.h"
#include "sperror.h"
#include "suffix.h"

int
BSIM3v1SmDelete(GENmodel **inModel, IFuid modname, GENmodel *kill)
{
BSIM3v1Smodel **model = (BSIM3v1Smodel**)inModel;
BSIM3v1Smodel *modfast = (BSIM3v1Smodel*)kill;
BSIM3v1Sinstance *here;
BSIM3v1Sinstance *prev = NULL;
BSIM3v1Smodel **oldmod;

    oldmod = model;
    for (; *model ; model = &((*model)->BSIM3v1SnextModel)) 
    {    if ((*model)->BSIM3v1SmodName == modname || 
             (modfast && *model == modfast))
	     goto delgot;
         oldmod = model;
    }
    return(E_NOMOD);

delgot:
    *oldmod = (*model)->BSIM3v1SnextModel; /* cut deleted device out of list */
    for (here = (*model)->BSIM3v1Sinstances; here; here = here->BSIM3v1SnextInstance)
    {    if(prev) FREE(prev);
         prev = here;
    }
    if(prev) FREE(prev);
    FREE(*model);
    return(OK);
}



