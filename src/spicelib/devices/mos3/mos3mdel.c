/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

#include "ngspice.h"
#include "mos3defs.h"
#include "sperror.h"
#include "suffix.h"


int
MOS3mDelete(GENmodel **inModel, IFuid modname, GENmodel *kill)
{
    MOS3model **model = (MOS3model **)inModel;
    MOS3model *modfast = (MOS3model *)kill;
    MOS3instance *here;
    MOS3instance *prev = NULL;
    MOS3model **oldmod;
    oldmod = model;
    for( ; *model ; model = &((*model)->MOS3nextModel)) {
        if( (*model)->MOS3modName == modname || 
                (modfast && *model == modfast) ) goto delgot;
        oldmod = model;
    }
    return(E_NOMOD);

delgot:
    *oldmod = (*model)->MOS3nextModel; /* cut deleted device out of list */
    for(here = (*model)->MOS3instances ; here ; here = here->MOS3nextInstance) {
        if(prev) FREE(prev);
        prev = here;
    }
    if(prev) FREE(prev);
    FREE(*model);
    return(OK);

}
