/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

#include "ngspice.h"
#include "vcvsdefs.h"
#include "suffix.h"


void
VCVSdestroy(GENmodel **inModel)
{
    VCVSmodel **model = (VCVSmodel **)inModel;
    VCVSinstance *here;
    VCVSinstance *prev = NULL;
    VCVSmodel *mod = *model;
    VCVSmodel *oldmod = NULL;

    for( ; mod ; mod = mod->VCVSnextModel) {
        if(oldmod) FREE(oldmod);
        oldmod = mod;
        prev = (VCVSinstance *)NULL;
        for(here = mod->VCVSinstances ; here ; here = here->VCVSnextInstance) {
            if(prev) FREE(prev);
            prev = here;
        }
        if(prev) FREE(prev);
    }
    if(oldmod) FREE(oldmod);
    *model = NULL;
}
