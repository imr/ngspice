/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Hong J. Park, Thomas L. Quarles
**********/
/*
 */

#include "ngspice.h"
#include "bsim1def.h"
#include "suffix.h"


void
B1destroy(GENmodel **inModel)
{

    B1model **model = (B1model**)inModel;
    B1instance *here;
    B1instance *prev = NULL;
    B1model *mod = *model;
    B1model *oldmod = NULL;

    for( ; mod ; mod = mod->B1nextModel) {
        if(oldmod) FREE(oldmod);
        oldmod = mod;
        prev = (B1instance *)NULL;
        for(here = mod->B1instances ; here ; here = here->B1nextInstance) {
            if(prev) FREE(prev);
            prev = here;
        }
        if(prev) FREE(prev);
    }
    if(oldmod) FREE(oldmod);
    *model = NULL;
}
