/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: Alan Gillespie
**********/
/*
 */

#include "ngspice.h"
#include "mos9defs.h"
#include "suffix.h"


void
MOS9destroy(GENmodel **inModel)
{
    MOS9model **model = (MOS9model **)inModel;
    MOS9instance *here;
    MOS9instance *prev = NULL;
    MOS9model *mod = *model;
    MOS9model *oldmod = NULL;

    for( ; mod ; mod = mod->MOS9nextModel) {
        if(oldmod) FREE(oldmod);
        oldmod = mod;
        prev = (MOS9instance *)NULL;
        for(here = mod->MOS9instances ; here ; here = here->MOS9nextInstance) {
            if(prev){
          if(prev->MOS9sens) FREE(prev->MOS9sens);
          FREE(prev);
            }
            prev = here;
        }
        if(prev) FREE(prev);
    }
    if(oldmod) FREE(oldmod);
    *model = NULL;
}
