/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

#include "ngspice.h"
#include "tradefs.h"
#include "sperror.h"
#include "suffix.h"


int
TRAmDelete(GENmodel **inModel, IFuid modname, GENmodel *kill)
{
    TRAmodel **model = (TRAmodel **)inModel;
    TRAmodel *modfast = (TRAmodel *)kill;
    TRAinstance *here;
    TRAinstance *prev = NULL;
    TRAmodel **oldmod;
    oldmod = model;
    for( ; *model ; model = &((*model)->TRAnextModel)) {
        if( (*model)->TRAmodName == modname || 
                (modfast && *model == modfast) ) goto delgot;
        oldmod = model;
    }
    return(E_NOMOD);

delgot:
    *oldmod = (*model)->TRAnextModel; /* cut deleted device out of list */
    for(here = (*model)->TRAinstances ; here ; here = here->TRAnextInstance) {
        if(prev) FREE(prev);
        prev = here;
    }
    if(prev) FREE(prev);
    FREE(*model);
    return(OK);

}
