/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Hong J. Park, Thomas L. Quarles
**********/
/*
 */

#include "ngspice.h"
#include "bsim1def.h"
#include "sperror.h"
#include "suffix.h"


int
B1mDelete(GENmodel **inModel, IFuid modname, GENmodel *kill)
{
    B1model **model = (B1model**)inModel;
    B1model *modfast = (B1model*)kill;
    B1instance *here;
    B1instance *prev = NULL;
    B1model **oldmod;
    oldmod = model;
    for( ; *model ; model = &((*model)->B1nextModel)) {
        if( (*model)->B1modName == modname || 
                (modfast && *model == modfast) ) goto delgot;
        oldmod = model;
    }
    return(E_NOMOD);

delgot:
    *oldmod = (*model)->B1nextModel; /* cut deleted device out of list */
    for(here = (*model)->B1instances ; here ; here = here->B1nextInstance) {
        if(prev) FREE(prev);
        prev = here;
    }
    if(prev) FREE(prev);
    FREE(*model);
    return(OK);

}
