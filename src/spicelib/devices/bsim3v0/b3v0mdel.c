/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1995 Min-Chie Jeng and Mansun Chan.
File: b3v0mdel.c
**********/

#include "ngspice.h"
#include "bsim3v0def.h"
#include "sperror.h"
#include "suffix.h"

int
BSIM3v0mDelete(GENmodel **inModel, IFuid modname, GENmodel *kill)
{
BSIM3v0model **model = (BSIM3v0model**)inModel;
BSIM3v0model *modfast = (BSIM3v0model*)kill;
BSIM3v0instance *here;
BSIM3v0instance *prev = NULL;
BSIM3v0model **oldmod;

    oldmod = model;
    for (; *model ; model = &((*model)->BSIM3v0nextModel)) 
    {    if ((*model)->BSIM3v0modName == modname || 
             (modfast && *model == modfast))
	     goto delgot;
         oldmod = model;
    }
    return(E_NOMOD);

delgot:
    *oldmod = (*model)->BSIM3v0nextModel; /* cut deleted device out of list */
    for (here = (*model)->BSIM3v0instances; here; here = here->BSIM3v0nextInstance)
    {    if(prev) FREE(prev);
         prev = here;
    }
    if(prev) FREE(prev);
    FREE(*model);
    return(OK);
}



