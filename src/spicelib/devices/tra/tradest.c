/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

#include "ngspice.h"
#include "tradefs.h"
#include "suffix.h"


void
TRAdestroy(GENmodel **inModel)
{
    TRAmodel **model = (TRAmodel **)inModel;
    TRAinstance *here;
    TRAinstance *prev = NULL;
    TRAmodel *mod = *model;
    TRAmodel *oldmod = NULL;

    for( ; mod ; mod = mod->TRAnextModel) {
        if(oldmod) FREE(oldmod);
        oldmod = mod;
        prev = (TRAinstance *)NULL;
        for(here = mod->TRAinstances ; here ; here = here->TRAnextInstance) {
            if(prev) FREE(prev);
            prev = here;
        }
        if(prev) FREE(prev);
    }
    if(oldmod) FREE(oldmod);
    *model = NULL;
}
