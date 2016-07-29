/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include "vcvsdefs.h"
#include "ngspice/suffix.h"


void
VCVSdestroy(GENmodel **inModel)
{
    VCVSmodel *mod = *(VCVSmodel **) inModel;

    while (mod) {
        VCVSmodel *next_mod = mod->VCVSnextModel;
        VCVSinstance *inst = mod->VCVSinstances;
        while (inst) {
            VCVSinstance *next_inst = inst->VCVSnextInstance;
            FREE(inst);
            inst = next_inst;
        }
        FREE(mod);
        mod = next_mod;
    }

    *inModel = NULL;
}
