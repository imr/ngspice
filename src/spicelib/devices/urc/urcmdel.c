/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1987 Thomas L. Quarles
**********/
/*
 */

#include "ngspice.h"
#include "urcdefs.h"
#include "sperror.h"
#include "suffix.h"


int
URCmDelete(GENmodel **inModel, IFuid modname, GENmodel *kill)
{
    URCmodel **model = (URCmodel**)inModel;
    URCmodel *modfast = (URCmodel *)kill;
    URCinstance *here;
    URCinstance *prev = NULL;
    URCmodel **oldmod;
    oldmod = model;
    for( ; *model ; model = &((*model)->URCnextModel)) {
        if( (*model)->URCmodName == modname || 
                (modfast && *model == modfast) ) goto delgot;
        oldmod = model;
    }
    return(E_NOMOD);

delgot:
    *oldmod = (*model)->URCnextModel; /* cut deleted device out of list */
    for(here = (*model)->URCinstances ; here ; here = here->URCnextInstance) {
        if(prev) FREE(prev);
        prev = here;
    }
    if(prev) FREE(prev);
    FREE(*model);
    return(OK);

}
