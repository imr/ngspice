/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Gordon Jacobs
**********/
/*
 */

#include "ngspice.h"
#include "swdefs.h"
#include "sperror.h"
#include "suffix.h"


int
SWmDelete(GENmodel **inModel, IFuid modname, GENmodel *kill)
{
    SWmodel **model = (SWmodel **)inModel;
    SWmodel *modfast = (SWmodel *)kill;
    SWinstance *here;
    SWinstance *prev = NULL;
    SWmodel **oldmod;
    oldmod = model;
    for( ; *model ; model = &((*model)->SWnextModel)) {
        if( (*model)->SWmodName == modname || 
                (modfast && *model == modfast) ) goto delgot;
        oldmod = model;
    }
    return(E_NOMOD);

delgot:
    *oldmod = (*model)->SWnextModel; /* cut deleted device out of list */
    for(here = (*model)->SWinstances ; here ; here = here->SWnextInstance) {
        if(prev) FREE(prev);
        prev = here;
    }
    if(prev) FREE(prev);
    FREE(*model);
    return(OK);

}
