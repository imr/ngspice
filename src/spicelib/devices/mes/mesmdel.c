/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 S. Hwang
**********/
/*
 */

#include "ngspice.h"
#include "mesdefs.h"
#include "sperror.h"
#include "suffix.h"


int
MESmDelete(GENmodel **inModel, IFuid modname, GENmodel *kill)
{
    MESmodel **model = (MESmodel**)inModel;
    MESmodel *modfast = (MESmodel*)kill;
    MESinstance *here;
    MESinstance *prev = NULL;
    MESmodel **oldmod;
    oldmod = model;
    for( ; *model ; model = &((*model)->MESnextModel)) {
        if( (*model)->MESmodName == modname || 
                (modfast && *model == modfast) ) goto delgot;
        oldmod = model;
    }
    return(E_NOMOD);

delgot:
    *oldmod = (*model)->MESnextModel; /* cut deleted device out of list */
    for(here = (*model)->MESinstances ; here ; here = here->MESnextInstance) {
        if(prev) FREE(prev);
        prev = here;
    }
    if(prev) FREE(prev);
    FREE(*model);
    return(OK);

}
