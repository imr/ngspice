/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

#include "ngspice.h"
#include "jfetdefs.h"
#include "sperror.h"
#include "suffix.h"


int
JFETmDelete(GENmodel **inModel, IFuid modname, GENmodel *kill)
{
    JFETmodel **model = (JFETmodel**)inModel;
    JFETmodel *modfast = (JFETmodel*)kill;
    JFETinstance *here;
    JFETinstance *prev = NULL;
    JFETmodel **oldmod;
    oldmod = model;
    for( ; *model ; model = &((*model)->JFETnextModel)) {
        if( (*model)->JFETmodName == modname || 
                (modfast && *model == modfast) ) goto delgot;
        oldmod = model;
    }
    return(E_NOMOD);

delgot:
    *oldmod = (*model)->JFETnextModel; /* cut deleted device out of list */
    for(here = (*model)->JFETinstances ; here ; here = here->JFETnextInstance) {
        if(prev) FREE(prev);
        prev = here;
    }
    if(prev) FREE(prev);
    FREE(*model);
    return(OK);

}
