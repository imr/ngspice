/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

#include "ngspice.h"
#include "mos2defs.h"
#include "suffix.h"


void
MOS2destroy(GENmodel **inModel)
{
    MOS2model **model = (MOS2model **)inModel;
    MOS2instance *here;
    MOS2instance *prev = NULL;
    MOS2model *mod = *model;
    MOS2model *oldmod = NULL;

    for( ; mod ; mod = mod->MOS2nextModel) {
        if(oldmod) FREE(oldmod);
        oldmod = mod;
        prev = (MOS2instance *)NULL;
        for(here = mod->MOS2instances ; here ; here = here->MOS2nextInstance) {
            if(prev){
          if(prev->MOS2sens) FREE(prev->MOS2sens);
          FREE(prev);
        }
            prev = here;
        }
        if(prev) FREE(prev);
    }
    if(oldmod) FREE(oldmod);
    *model = NULL;
}
