/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

#include "ngspice.h"
#include "mos3defs.h"
#include "suffix.h"


void
MOS3destroy(GENmodel **inModel)
{
    MOS3model **model = (MOS3model **)inModel;
    MOS3instance *here;
    MOS3instance *prev = NULL;
    MOS3model *mod = *model;
    MOS3model *oldmod = NULL;

    for( ; mod ; mod = mod->MOS3nextModel) {
        if(oldmod) FREE(oldmod);
        oldmod = mod;
        prev = (MOS3instance *)NULL;
        for(here = mod->MOS3instances ; here ; here = here->MOS3nextInstance) {
            if(prev){
          if(prev->MOS3sens) FREE(prev->MOS3sens);
          FREE(prev);
            }
            prev = here;
        }
        if(prev) FREE(prev);
    }
    if(oldmod) FREE(oldmod);
    *model = NULL;
}
