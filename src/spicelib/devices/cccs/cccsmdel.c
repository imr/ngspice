/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

#include "ngspice.h"
#include "cccsdefs.h"
#include "sperror.h"
#include "suffix.h"


int
CCCSmDelete(GENmodel **inModel, IFuid modname, GENmodel *kill)
{

    CCCSmodel **model = (CCCSmodel**)inModel;
    CCCSmodel *modfast = (CCCSmodel*)kill;
    CCCSinstance *here;
    CCCSinstance *prev = NULL;
    CCCSmodel **oldmod;
    oldmod = model;
    for( ; *model ; model = &((*model)->CCCSnextModel)) {
        if( (*model)->CCCSmodName == modname || 
                (modfast && *model == modfast) ) goto delgot;
        oldmod = model;
    }
    return(E_NOMOD);

delgot:
    *oldmod = (*model)->CCCSnextModel; /* cut deleted device out of list */
    for(here = (*model)->CCCSinstances ; here ; here = here->CCCSnextInstance) {
        if(prev) FREE(prev);
        prev = here;
    }
    if(prev) FREE(prev);
    FREE(*model);
    return(OK);

}
