/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

#include "ngspice.h"
#include "inddefs.h"
#include "suffix.h"


void
INDdestroy(GENmodel **inModel)
{
    INDmodel **model = (INDmodel**)inModel;
    INDinstance *here;
    INDinstance *prev = NULL;
    INDmodel *mod = *model;
    INDmodel *oldmod = NULL;

    for( ; mod ; mod = mod->INDnextModel) {
        if(oldmod) FREE(oldmod);
        oldmod = mod;
        prev = (INDinstance *)NULL;
        for(here = mod->INDinstances ; here ; here = here->INDnextInstance) {
            if(prev) FREE(prev);
            prev = here;
        }
        if(prev) FREE(prev);
    }
    if(oldmod) FREE(oldmod);
    *model = NULL;
}
