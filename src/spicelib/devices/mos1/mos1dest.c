/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

#include "ngspice.h"
#include "mos1defs.h"
#include "suffix.h"


void
MOS1destroy(GENmodel **inModel)
{
    MOS1model **model = (MOS1model**)inModel;
    MOS1instance *here;
    MOS1instance *prev = NULL;
    MOS1model *mod = *model;
    MOS1model *oldmod = NULL;

    for( ; mod ; mod = mod->MOS1nextModel) {
        if(oldmod) FREE(oldmod);
        oldmod = mod;
        prev = (MOS1instance *)NULL;
        for(here = mod->MOS1instances ; here ; here = here->MOS1nextInstance) {
            if(prev){
                if(prev->MOS1sens) FREE(prev->MOS1sens); 
                FREE(prev);
            }
            prev = here;
        }
        if(prev) FREE(prev);
    }
    if(oldmod) FREE(oldmod);
    *model = NULL;
}
