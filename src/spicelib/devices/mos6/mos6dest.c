/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1989 Takayasu Sakurai
**********/
/*
 */

#include "ngspice.h"
#include "mos6defs.h"
#include "suffix.h"


void
MOS6destroy(GENmodel **inModel)
{
    MOS6model **model = (MOS6model**)inModel;
    MOS6instance *here;
    MOS6instance *prev = NULL;
    MOS6model *mod = *model;
    MOS6model *oldmod = NULL;

    for( ; mod ; mod = mod->MOS6nextModel) {
        if(oldmod) FREE(oldmod);
        oldmod = mod;
        prev = (MOS6instance *)NULL;
        for(here = mod->MOS6instances ; here ; here = here->MOS6nextInstance) {
            if(prev){
                if(prev->MOS6sens) FREE(prev->MOS6sens); 
                FREE(prev);
            }
            prev = here;
        }
        if(prev) FREE(prev);
    }
    if(oldmod) FREE(oldmod);
    *model = NULL;
}
