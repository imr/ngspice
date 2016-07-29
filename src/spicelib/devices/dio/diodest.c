/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include "diodefs.h"
#include "ngspice/suffix.h"


void
DIOdestroy(GENmodel **inModel)
{
    DIOmodel *mod = *(DIOmodel**) inModel;

    while (mod) {
        DIOmodel *next_mod = mod->DIOnextModel;
        DIOinstance *inst = mod->DIOinstances;
        while (inst) {
            DIOinstance *next_inst = inst->DIOnextInstance;
            FREE(inst);
            inst = next_inst;
        }
        FREE(mod);
        mod = next_mod;
    }

    *inModel = NULL;
}
