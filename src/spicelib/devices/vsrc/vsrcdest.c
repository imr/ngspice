/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

#include "ngspice.h"
#include "vsrcdefs.h"
#include "suffix.h"


void
VSRCdestroy(GENmodel **inModel)
{
    VSRCmodel **model = (VSRCmodel**)inModel;
    VSRCinstance *here;
    VSRCinstance *prev = NULL;
    VSRCmodel *mod = *model;
    VSRCmodel *oldmod = NULL;

    for( ; mod ; mod = mod->VSRCnextModel) {
        if(oldmod) FREE(oldmod);
        oldmod = mod;
        prev = (VSRCinstance *)NULL;
        for(here = mod->VSRCinstances ; here ; here = here->VSRCnextInstance) {
            if(prev) FREE(prev);
            prev = here;
        }
        if(prev) FREE(prev);
    }
    if(oldmod) FREE(oldmod);
    *model = NULL;
}
