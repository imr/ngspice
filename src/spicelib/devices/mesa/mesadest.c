/**********
Copyright 1993: T. Ytterdal, K. Lee, M. Shur and T. A. Fjeldly. All rights reserved.
Author: Trond Ytterdal
**********/

#include "ngspice.h"
#include "mesadefs.h"
#include "suffix.h"


void
MESAdestroy(GENmodel **inModel)
{
    MESAmodel **model = (MESAmodel**)inModel;
    MESAinstance *here;
    MESAinstance *prev = NULL;
    MESAmodel *mod = *model;
    MESAmodel *oldmod = NULL;

    for( ; mod ; mod = mod->MESAnextModel) {
        if(oldmod) FREE(oldmod);
        oldmod = mod;
        prev = (MESAinstance *)NULL;
        for(here = mod->MESAinstances ; here ; here = here->MESAnextInstance) {
            if(prev) FREE(prev);
            prev = here;
        }
        if(prev) FREE(prev);
    }
    if(oldmod) FREE(oldmod);
    *model = NULL;
    return;
}
