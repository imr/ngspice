/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Hong J. Park, Thomas L. Quarles
**********/
/*
 */

#include "ngspice.h"
#include "bsim2def.h"
#include "sperror.h"
#include "suffix.h"


int
B2mDelete(GENmodel **inModel, IFuid modname, GENmodel *kill)
{
    B2model **model = (B2model**)inModel;
    B2model *modfast = (B2model*)kill;
    B2instance *here;
    B2instance *prev = NULL;
    B2model **oldmod;
    oldmod = model;
    for( ; *model ; model = &((*model)->B2nextModel)) {
        if( (*model)->B2modName == modname || 
                (modfast && *model == modfast) ) goto delgot;
        oldmod = model;
    }
    return(E_NOMOD);

delgot:
    *oldmod = (*model)->B2nextModel; /* cut deleted device out of list */
    for(here = (*model)->B2instances ; here ; here = here->B2nextInstance) {
        if(prev) FREE(prev);
        prev = here;
    }
    if(prev) FREE(prev);
    FREE(*model);
    return(OK);

}

