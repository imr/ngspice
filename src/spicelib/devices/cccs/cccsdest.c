/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

#include "ngspice.h"
#include "cccsdefs.h"
#include "suffix.h"


void
CCCSdestroy(GENmodel **inModel)

{
    CCCSmodel **model = (CCCSmodel**)inModel;
    CCCSinstance *here;
    CCCSinstance *prev = NULL;
    CCCSmodel *mod = *model;
    CCCSmodel *oldmod = NULL;

    for( ; mod ; mod = mod->CCCSnextModel) {
        if(oldmod) FREE(oldmod);
        oldmod = mod;
        prev = (CCCSinstance *)NULL;
        for(here = mod->CCCSinstances ; here ; here = here->CCCSnextInstance) {
            if(prev) FREE(prev);
            prev = here;
        }
        if(prev) FREE(prev);
    }
    if(oldmod) FREE(oldmod);
    *model = NULL;
}
