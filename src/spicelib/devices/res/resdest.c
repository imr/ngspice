/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: Apr 2000 - Paolo Nenzi
**********/

#include "ngspice/ngspice.h"
#include "resdefs.h"


void
RESdestroy(GENmodel **inModel)
{
    RESmodel *mod = *(RESmodel **) inModel;

    while (mod) {
        RESmodel *next_mod = mod->RESnextModel;
        RESinstance *inst = mod->RESinstances;
        while (inst) {
            RESinstance *next_inst = inst->RESnextInstance;
            FREE(inst);
            inst = next_inst;
        }
        FREE(mod);
        mod = next_mod;
    }

    *inModel = NULL;
}
