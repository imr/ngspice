/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: Alan Gillespie
**********/
/*
 */

/*
 * This routine deletes all BJT2s from the circuit and frees
 * all storage they were using.
 */

#include "ngspice.h"
#include "bjt2defs.h"
#include "suffix.h"


void
BJT2destroy(GENmodel **inModel)
{
    BJT2model **model = (BJT2model**)inModel;
    BJT2instance *here;
    BJT2instance *prev = NULL;
    BJT2model *mod = *model;
    BJT2model *oldmod = NULL;

    for( ; mod ; mod = mod->BJT2nextModel) {
        if(oldmod) FREE(oldmod);
        oldmod = mod;
        prev = (BJT2instance *)NULL;
        for(here = mod->BJT2instances ; here ; here = here->BJT2nextInstance) {
            if(prev){
                if(prev->BJT2sens) FREE(prev->BJT2sens);
                FREE(prev);
            }
            prev = here;
        }
        if(prev){
            if(prev->BJT2sens) FREE(prev->BJT2sens);
            FREE(prev);
        }
    }
    if(oldmod) FREE(oldmod);
    *model = NULL;
}
