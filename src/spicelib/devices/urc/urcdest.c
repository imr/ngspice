/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1987 Thomas L. Quarles
**********/
/*
 */


#include "ngspice.h"
#include "urcdefs.h"
#include "suffix.h"


void
URCdestroy(GENmodel **inModel)
{
    URCmodel **model = (URCmodel **)inModel;
    URCinstance *here;
    URCinstance *prev = NULL;
    URCmodel *mod = *model;
    URCmodel *oldmod = NULL;

    for( ; mod ; mod = mod->URCnextModel) {
        if(oldmod) FREE(oldmod);
        oldmod = mod;
        prev = (URCinstance *)NULL;
        for(here = mod->URCinstances ; here ; here = here->URCnextInstance) {
            if(prev) FREE(prev);
            prev = here;
        }
        if(prev) FREE(prev);
    }
    if(oldmod) FREE(oldmod);
    *model = NULL;
}
