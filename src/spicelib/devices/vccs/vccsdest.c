/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

#include "ngspice.h"
#include "vccsdefs.h"
#include "suffix.h"


void
VCCSdestroy(GENmodel **inModel)
{
    VCCSmodel **model = (VCCSmodel**)inModel;
    VCCSinstance *here;
    VCCSinstance *prev = NULL;
    VCCSmodel *mod = *model;
    VCCSmodel *oldmod = NULL;

    for( ; mod ; mod = mod->VCCSnextModel) {
        if(oldmod) FREE(oldmod);
        oldmod = mod;
        prev = (VCCSinstance *)NULL;
        for(here = mod->VCCSinstances ; here ; here = here->VCCSnextInstance) {
            if(prev) FREE(prev);
            prev = here;
        }
        if(prev) FREE(prev);
    }
    if(oldmod) FREE(oldmod);
    *model = NULL;
}
