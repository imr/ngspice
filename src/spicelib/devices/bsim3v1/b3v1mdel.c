/**********
 * Copyright 1990 Regents of the University of California. All rights reserved.
 * File: b3v1mdel.c
 * Author: 1995 Min-Chie Jeng and Mansun Chan. 
 * Modified by Paolo Nenzi 2002
 **********/
 
/* 
 * Release Notes: 
 * BSIM3v3.1,   Released by yuhua  96/12/08
 */

#include "ngspice.h"
#include "bsim3v1def.h"
#include "sperror.h"
#include "suffix.h"

int
BSIM3v1mDelete(GENmodel **inModel, IFuid modname, GENmodel *kill)
{
BSIM3v1model **model = (BSIM3v1model**)inModel;
BSIM3v1model *modfast = (BSIM3v1model*)kill;
BSIM3v1instance *here;
BSIM3v1instance *prev = NULL;
BSIM3v1model **oldmod;

    oldmod = model;
    for (; *model ; model = &((*model)->BSIM3v1nextModel)) 
    {    if ((*model)->BSIM3v1modName == modname || 
             (modfast && *model == modfast))
	     goto delgot;
         oldmod = model;
    }
    return(E_NOMOD);

delgot:
    *oldmod = (*model)->BSIM3v1nextModel; /* cut deleted device out of list */
    for (here = (*model)->BSIM3v1instances; here; here = here->BSIM3v1nextInstance)
    {    if(prev) FREE(prev);
         prev = here;
    }
    if(prev) FREE(prev);
    FREE(*model);
    return(OK);
}



