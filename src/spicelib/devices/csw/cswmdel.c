/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Gordon Jacobs
**********/
/*
 */

#include "ngspice.h"
#include "cswdefs.h"
#include "sperror.h"
#include "suffix.h"


int
CSWmDelete(GENmodel **inModel, IFuid modname, GENmodel *kill)
{
    CSWmodel **model = (CSWmodel**)inModel;
    CSWmodel *modfast = (CSWmodel*)kill;
    CSWinstance *here;
    CSWinstance *prev = NULL;
    CSWmodel **oldmod;
    oldmod = model;
    for( ; *model ; model = &((*model)->CSWnextModel)) {
        if( (*model)->CSWmodName == modname || 
                (modfast && *model == modfast) ) goto delgot;
        oldmod = model;
    }
    return(E_NOMOD);

delgot:
    *oldmod = (*model)->CSWnextModel; /* cut deleted device out of list */
    for(here = (*model)->CSWinstances ; here ; here = here->CSWnextInstance) {
        if(prev) FREE(prev);
        prev = here;
    }
    if(prev) FREE(prev);
    FREE(*model);
    return(OK);

}
