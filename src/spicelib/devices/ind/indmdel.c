/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

#include "ngspice.h"
#include "inddefs.h"
#include "sperror.h"
#include "suffix.h"


int
INDmDelete(GENmodel **inModel, IFuid modname, GENmodel *kill)
{
    INDmodel **model = (INDmodel**)inModel;
    INDmodel *modfast = (INDmodel*)kill;
    INDinstance *here;
    INDinstance *prev = NULL;
    INDmodel **oldmod;
    oldmod = model;
    for( ; *model ; model = &((*model)->INDnextModel)) {
        if( (*model)->INDmodName == modname || 
                (modfast && *model == modfast) ) goto delgot;
        oldmod = model;
    }
    return(E_NOMOD);

delgot:
    *oldmod = (*model)->INDnextModel; /* cut deleted device out of list */
    for(here = (*model)->INDinstances ; here ; here = here->INDnextInstance) {
        if(prev) FREE(prev);
        prev = here;
    }
    if(prev) FREE(prev);
    FREE(*model);
    return(OK);

}
