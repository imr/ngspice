/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

#include "ngspice.h"
#include "mos2defs.h"
#include "sperror.h"
#include "suffix.h"


int
MOS2mDelete(GENmodel **inModel, IFuid modname, GENmodel *kill)
{
    MOS2model **model = (MOS2model **)inModel;
    MOS2model *modfast = (MOS2model *)kill;
    MOS2instance *here;
    MOS2instance *prev = NULL;
    MOS2model **oldmod;
    oldmod = model;
    for( ; *model ; model = &((*model)->MOS2nextModel)) {
        if( (*model)->MOS2modName == modname || 
                (modfast && *model == modfast) ) goto delgot;
        oldmod = model;
    }
    return(E_NOMOD);

delgot:
    *oldmod = (*model)->MOS2nextModel; /* cut deleted device out of list */
    for(here = (*model)->MOS2instances ; here ; here = here->MOS2nextInstance) {
        if(prev) FREE(prev);
        prev = here;
    }
    if(prev) FREE(prev);
    FREE(*model);
    return(OK);

}
