/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1995 Min-Chie Jeng and Mansun Chan.
File: b3v2mdel.c
**********/

#include "ngspice.h"
#include <stdio.h>
#include "bsim3v2def.h"
#include "sperror.h"
#include "suffix.h"

int
BSIM3V2mDelete(inModel,modname,kill)
GENmodel **inModel;
IFuid modname;
GENmodel *kill;
{
BSIM3V2model **model = (BSIM3V2model**)inModel;
BSIM3V2model *modfast = (BSIM3V2model*)kill;
BSIM3V2instance *here;
BSIM3V2instance *prev = NULL;
BSIM3V2model **oldmod;

    oldmod = model;
    for (; *model ; model = &((*model)->BSIM3V2nextModel)) 
    {    if ((*model)->BSIM3V2modName == modname || 
             (modfast && *model == modfast))
	     goto delgot;
         oldmod = model;
    }
    return(E_NOMOD);

delgot:
    *oldmod = (*model)->BSIM3V2nextModel; /* cut deleted device out of list */
    for (here = (*model)->BSIM3V2instances; here; here = here->BSIM3V2nextInstance)
    {    if(prev) FREE(prev);
         prev = here;
    }
    if(prev) FREE(prev);
    FREE(*model);
    return(OK);
}



