/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Hong J. Park, Thomas L. Quarles
**********/
/*
 */

#include "ngspice.h"
#include "bsim2def.h"
#include "suffix.h"


void
B2destroy(GENmodel **inModel)
{

    B2model **model = (B2model**)inModel;
    B2instance *here;
    B2instance *prev = NULL;
    B2model *mod = *model;
    B2model *oldmod = NULL;

    for( ; mod ; mod = mod->B2nextModel) {
        if(oldmod) FREE(oldmod);
        oldmod = mod;
        prev = (B2instance *)NULL;
        for(here = mod->B2instances ; here ; here = here->B2nextInstance) {
            if(prev) FREE(prev);
            prev = here;
        }
        if(prev) FREE(prev);
    }
    if(oldmod) FREE(oldmod);
    *model = NULL;
}

