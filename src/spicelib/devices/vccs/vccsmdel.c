/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

#include "ngspice.h"
#include "vccsdefs.h"
#include "sperror.h"
#include "suffix.h"


int
VCCSmDelete(GENmodel **inModel, IFuid modname, GENmodel *kill)
{
    VCCSmodel **model = (VCCSmodel **)inModel;
    VCCSmodel *modfast = (VCCSmodel *)kill;
    VCCSinstance *here;
    VCCSinstance *prev = NULL;
    VCCSmodel **oldmod;
    oldmod = model;
    for( ; *model ; model = &((*model)->VCCSnextModel)) {
        if( (*model)->VCCSmodName == modname || 
                (modfast && *model == modfast) ) goto delgot;
        oldmod = model;
    }
    return(E_NOMOD);

delgot:
    *oldmod = (*model)->VCCSnextModel; /* cut deleted device out of list */
    for(here = (*model)->VCCSinstances ; here ; here = here->VCCSnextInstance) {
        if(prev) FREE(prev);
        prev = here;
    }
    if(prev) FREE(prev);
    FREE(*model);
    return(OK);

}
