/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

#include "ngspice.h"
#include "vcvsdefs.h"
#include "sperror.h"
#include "suffix.h"


int
VCVSmDelete(GENmodel **inModel, IFuid modname, GENmodel *kill)
{
    VCVSmodel **model = (VCVSmodel**)inModel;
    VCVSmodel *modfast = (VCVSmodel *)kill;
    VCVSinstance *here;
    VCVSinstance *prev = NULL;
    VCVSmodel **oldmod;
    oldmod = model;
    for( ; *model ; model = &((*model)->VCVSnextModel)) {
        if( (*model)->VCVSmodName == modname || 
                (modfast && *model == modfast) ) goto delgot;
        oldmod = model;
    }
    return(E_NOMOD);

delgot:
    *oldmod = (*model)->VCVSnextModel; /* cut deleted device out of list */
    for(here = (*model)->VCVSinstances ; here ; here = here->VCVSnextInstance) {
        if(prev) FREE(prev);
        prev = here;
    }
    if(prev) FREE(prev);
    FREE(*model);
    return(OK);

}
