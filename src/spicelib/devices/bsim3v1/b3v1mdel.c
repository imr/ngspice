/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1995 Min-Chie Jeng and Mansun Chan.
File: b3v1mdel.c
**********/

#include "ngspice.h"
#include <stdio.h>
#include "bsim3v1def.h"
#include "sperror.h"
#include "suffix.h"

int
BSIM3V1mDelete(inModel,modname,kill)
GENmodel **inModel;
IFuid modname;
GENmodel *kill;
{
BSIM3V1model **model = (BSIM3V1model**)inModel;
BSIM3V1model *modfast = (BSIM3V1model*)kill;
BSIM3V1instance *here;
BSIM3V1instance *prev = NULL;
BSIM3V1model **oldmod;

    oldmod = model;
    for (; *model ; model = &((*model)->BSIM3V1nextModel)) 
    {    if ((*model)->BSIM3V1modName == modname || 
             (modfast && *model == modfast))
	     goto delgot;
         oldmod = model;
    }
    return(E_NOMOD);

delgot:
    *oldmod = (*model)->BSIM3V1nextModel; /* cut deleted device out of list */
    for (here = (*model)->BSIM3V1instances; here; here = here->BSIM3V1nextInstance)
    {    if(prev) FREE(prev);
         prev = here;
    }
    if(prev) FREE(prev);
    FREE(*model);
    return(OK);
}



