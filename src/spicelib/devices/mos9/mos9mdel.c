/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: Alan Gillespie
**********/
/*
 */

#include "ngspice.h"
#include "mos9defs.h"
#include "sperror.h"
#include "suffix.h"


int
MOS9mDelete(GENmodel **inModel, IFuid modname, GENmodel *kill)
{
    MOS9model **model = (MOS9model **)inModel;
    MOS9model *modfast = (MOS9model *)kill;
    MOS9instance *here;
    MOS9instance *prev = NULL;
    MOS9model **oldmod;
    oldmod = model;
    for( ; *model ; model = &((*model)->MOS9nextModel)) {
        if( (*model)->MOS9modName == modname || 
                (modfast && *model == modfast) ) goto delgot;
        oldmod = model;
    }
    return(E_NOMOD);

delgot:
    *oldmod = (*model)->MOS9nextModel; /* cut deleted device out of list */
    for(here = (*model)->MOS9instances ; here ; here = here->MOS9nextInstance) {
        if(prev) FREE(prev);
        prev = here;
    }
    if(prev) FREE(prev);
    FREE(*model);
    return(OK);

}
