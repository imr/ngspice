/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

#include "ngspice.h"
#include "ccvsdefs.h"
#include "suffix.h"


void
CCVSdestroy(GENmodel **inModel)
{
    CCVSmodel **model = (CCVSmodel**)inModel;
    CCVSinstance *here;
    CCVSinstance *prev = NULL;
    CCVSmodel *mod = *model;
    CCVSmodel *oldmod = NULL;

    for( ; mod ; mod = mod->CCVSnextModel) {
        if(oldmod) FREE(oldmod);
        oldmod = mod;
        prev = (CCVSinstance *)NULL;
        for(here = mod->CCVSinstances ; here ; here = here->CCVSnextInstance) {
            if(prev) FREE(prev);
            prev = here;
        }
        if(prev) FREE(prev);
    }
    if(oldmod) FREE(oldmod);
    *model = NULL;
}
