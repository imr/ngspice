/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 S. Hwang
**********/
/*
 Imported into mesa model: 2001 Paolo Nenzi
 */

#include "ngspice.h"
#include "mesadefs.h"
#include "sperror.h"
#include "suffix.h"


int
MESAmDelete(GENmodel **inModel, IFuid modname, GENmodel *kill)
{
    MESAmodel **model = (MESAmodel**)inModel;
    MESAmodel *modfast = (MESAmodel*)kill;
    MESAinstance *here;
    MESAinstance *prev = NULL;
    MESAmodel **oldmod;
    oldmod = model;
    for( ; *model ; model = &((*model)->MESAnextModel)) {
        if( (*model)->MESAmodName == modname || 
                (modfast && *model == modfast) ) goto delgot;
        oldmod = model;
    }
    return(E_NOMOD);

delgot:
    *oldmod = (*model)->MESAnextModel; /* cut deleted device out of list */
    for(here = (*model)->MESAinstances ; here ; here = here->MESAnextInstance) {
        if(prev) FREE(prev);
        prev = here;
    }
    if(prev) FREE(prev);
    FREE(*model);
    return(OK);

}
