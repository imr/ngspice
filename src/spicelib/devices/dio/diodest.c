/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

#include "ngspice.h"
#include "diodefs.h"
#include "suffix.h"


void
DIOdestroy(GENmodel **inModel)
{
    DIOmodel **model = (DIOmodel**)inModel;
    DIOinstance *here;
    DIOinstance *prev = NULL;
    DIOmodel *mod = *model;
    DIOmodel *oldmod = NULL;

    for( ; mod ; mod = mod->DIOnextModel) {
        if(oldmod) FREE(oldmod);
        oldmod = mod;
        prev = (DIOinstance *)NULL;
        for(here = mod->DIOinstances ; here ; here = here->DIOnextInstance) {
            if(prev) FREE(prev);
            prev = here;
        }
        if(prev) FREE(prev);
    }
    if(oldmod) FREE(oldmod);
    *model = NULL;
}
