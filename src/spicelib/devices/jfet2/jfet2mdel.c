/**********
based on jfetmdel.c
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles

Modified to jfet2 for PS model definition ( Anthony E. Parker )
   Copyright 1994  Macquarie University, Sydney Australia.
**********/
/*
 */

#include "ngspice.h"
#include "jfet2defs.h"
#include "sperror.h"
#include "suffix.h"


int
JFET2mDelete(GENmodel **inModel, IFuid modname, GENmodel *kill)
{
    JFET2model **model = (JFET2model**)inModel;
    JFET2model *modfast = (JFET2model*)kill;
    JFET2instance *here;
    JFET2instance *prev = NULL;
    JFET2model **oldmod;
    oldmod = model;
    for( ; *model ; model = &((*model)->JFET2nextModel)) {
        if( (*model)->JFET2modName == modname || 
                (modfast && *model == modfast) ) goto delgot;
        oldmod = model;
    }
    return(E_NOMOD);

delgot:
    *oldmod = (*model)->JFET2nextModel; /* cut deleted device out of list */
    for(here = (*model)->JFET2instances ; here ; here = here->JFET2nextInstance) {
        if(prev) FREE(prev);
        prev = here;
    }
    if(prev) FREE(prev);
    FREE(*model);
    return(OK);

}
