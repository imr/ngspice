/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 S. Hwang
**********/
/*
Imported into hfeta model: Paolo Nenzi 2001
 */

#include "ngspice.h"
#include "hfetdefs.h"
#include "sperror.h"
#include "suffix.h"


int
HFETAmDelete(GENmodel **inModel, IFuid modname, GENmodel *kill)
{
    HFETAmodel **model = (HFETAmodel**)inModel;
    HFETAmodel *modfast = (HFETAmodel*)kill;
    HFETAinstance *here;
    HFETAinstance *prev = NULL;
    HFETAmodel **oldmod;
    oldmod = model;
    for( ; *model ; model = &((*model)->HFETAnextModel)) {
        if( (*model)->HFETAmodName == modname || 
                (modfast && *model == modfast) ) goto delgot;
        oldmod = model;
    }
    return(E_NOMOD);

delgot:
    *oldmod = (*model)->HFETAnextModel; /* cut deleted device out of list */
    for(here = (*model)->HFETAinstances ; here ; here = here->HFETAnextInstance) {
        if(prev) FREE(prev);
        prev = here;
    }
    if(prev) FREE(prev);
    FREE(*model);
    return(OK);

}
