/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: Apr 2000 - Paolo Nenzi
**********/
/*
 */

#include "ngspice.h"
#include "resdefs.h"


void
RESdestroy(GENmodel **inModel)
{
    RESmodel **model = (RESmodel **)inModel;
    RESinstance *here;
    RESinstance *prev = NULL;
    RESmodel *mod = *model;
    RESmodel *oldmod = NULL;

    for( ; mod ; mod = mod->RESnextModel) {
        if(oldmod) FREE(oldmod);
        oldmod = mod;
        prev = (RESinstance *)NULL;
        for(here = mod->RESinstances ; here ; here = here->RESnextInstance) {
            if(prev) FREE(prev);
            prev = here;
        }
        if(prev) FREE(prev);
    }
    if(oldmod) FREE(oldmod);
    *model = NULL;
}
