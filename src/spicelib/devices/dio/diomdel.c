/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

#include "ngspice.h"
#include "diodefs.h"
#include "sperror.h"
#include "suffix.h"


int
DIOmDelete(GENmodel **inModel, IFuid modname, GENmodel *kill)
{
    DIOmodel **model = (DIOmodel**)inModel;
    DIOmodel *modfast = (DIOmodel*)kill;
    DIOinstance *here;
    DIOinstance *prev = NULL;
    DIOmodel **oldmod;
    oldmod = model;
    for( ; *model ; model = &((*model)->DIOnextModel)) {
        if( (*model)->DIOmodName == modname || 
                (modfast && *model == modfast) ) goto delgot;
        oldmod = model;
    }
    return(E_NOMOD);

delgot:
    *oldmod = (*model)->DIOnextModel; /* cut deleted device out of list */
    for(here = (*model)->DIOinstances ; here ; here = here->DIOnextInstance) {
        if(prev) FREE(prev);
        prev = here;
    }
    if(prev) FREE(prev);
    FREE(*model);
    return(OK);

}
