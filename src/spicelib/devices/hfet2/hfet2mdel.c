/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 S. Hwang
**********/
/*
Imported into hfet2 model: Paolo Nenzi 2001
 */

#include "ngspice.h"
#include "hfet2defs.h"
#include "sperror.h"
#include "suffix.h"


int
HFET2mDelete(GENmodel **inModel, IFuid modname, GENmodel *kill)
{
    HFET2model **model = (HFET2model**)inModel;
    HFET2model *modfast = (HFET2model*)kill;
    HFET2instance *here;
    HFET2instance *prev = NULL;
    HFET2model **oldmod;
    oldmod = model;
    for( ; *model ; model = &((*model)->HFET2nextModel)) {
        if( (*model)->HFET2modName == modname || 
                (modfast && *model == modfast) ) goto delgot;
        oldmod = model;
    }
    return(E_NOMOD);

delgot:
    *oldmod = (*model)->HFET2nextModel; /* cut deleted device out of list */
    for(here = (*model)->HFET2instances ; here ; here = here->HFET2nextInstance) {
        if(prev) FREE(prev);
        prev = here;
    }
    if(prev) FREE(prev);
    FREE(*model);
    return(OK);

}
