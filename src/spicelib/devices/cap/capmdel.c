/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

#include "ngspice.h"
#include "capdefs.h"
#include "sperror.h"
#include "suffix.h"


int
CAPmDelete(inModel,modname,kill)
    GENmodel **inModel;
    IFuid modname;
    GENmodel *kill;

{

    CAPmodel *modfast = (CAPmodel*)kill;
    CAPmodel **model = (CAPmodel**)inModel;
    CAPinstance *here;
    CAPinstance *prev = NULL;
    CAPmodel **oldmod;
    oldmod = model;
    for( ; *model ; model = &((*model)->CAPnextModel)) {
        if( (*model)->CAPmodName == modname || 
                (modfast && *model == modfast) ) goto delgot;
        oldmod = model;
    }
    return(E_NOMOD);

delgot:
    *oldmod = (*model)->CAPnextModel; /* cut deleted device out of list */
    for(here = (*model)->CAPinstances ; here ; here = here->CAPnextInstance) {
        if(prev) FREE(prev);
        prev = here;
    }
    if(prev) FREE(prev);
    FREE(*model);
    return(OK);

}

