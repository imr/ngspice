/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 S. Hwang
**********/
/*
 */

#include "ngspice.h"
#include "mesdefs.h"
#include "suffix.h"


void
MESdestroy(GENmodel **inModel)
{
    MESmodel **model = (MESmodel**)inModel;
    MESinstance *here;
    MESinstance *prev = NULL;
    MESmodel *mod = *model;
    MESmodel *oldmod = NULL;

    for( ; mod ; mod = mod->MESnextModel) {
        if(oldmod) FREE(oldmod);
        oldmod = mod;
        prev = (MESinstance *)NULL;
        for(here = mod->MESinstances ; here ; here = here->MESnextInstance) {
            if(prev) FREE(prev);
            prev = here;
        }
        if(prev) FREE(prev);
    }
    if(oldmod) FREE(oldmod);
    *model = NULL;
}
