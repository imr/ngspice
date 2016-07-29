/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include "ccvsdefs.h"
#include "ngspice/suffix.h"


void
CCVSdestroy(GENmodel **inModel)
{
    CCVSmodel *mod = *(CCVSmodel**) inModel;

    while (mod) {
        CCVSmodel *next_mod = mod->CCVSnextModel;
        CCVSinstance *inst = mod->CCVSinstances;
        while (inst) {
            CCVSinstance *next_inst = inst->CCVSnextInstance;
            FREE(inst);
            inst = next_inst;
        }
        FREE(mod);
        mod = next_mod;
    }

    *inModel = NULL;
}
