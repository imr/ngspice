/**** BSIM3v3.2.4, Released by Xuemei Xi 12/21/2001 ****/

/**********
 * Copyright 2001 Regents of the University of California. All rights reserved.
 * File: b3mdel.c of BSIM3v3.2.4
 * Author: 1995 Min-Chie Jeng and Mansun Chan. 
 * Author: 1997-1999 Weidong Liu.
 * Author: 2001  Xuemei Xi
 * Modified by Paolo Nenzi 2002
 **********/

#include "ngspice.h"
#include "bsim3def.h"
#include "sperror.h"
#include "suffix.h"

int
BSIM3mDelete (GENmodel **inModel, IFuid modname, GENmodel *kill)
{
BSIM3model **model = (BSIM3model**)inModel;
BSIM3model *modfast = (BSIM3model*)kill;
BSIM3instance *here;
BSIM3instance *prev = NULL;
BSIM3model **oldmod;

    oldmod = model;
    for (; *model ; model = &((*model)->BSIM3nextModel)) 
    {    if ((*model)->BSIM3modName == modname || 
             (modfast && *model == modfast))
	     goto delgot;
         oldmod = model;
    }
    return(E_NOMOD);

delgot:
    *oldmod = (*model)->BSIM3nextModel; /* cut deleted device out of list */
    for (here = (*model)->BSIM3instances; here; here = here->BSIM3nextInstance)
    {    if(prev) FREE(prev);
         prev = here;
    }
    if(prev) FREE(prev);
    FREE(*model);
    return(OK);
}



