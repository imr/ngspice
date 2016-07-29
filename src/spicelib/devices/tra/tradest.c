/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include "tradefs.h"
#include "ngspice/suffix.h"


void
TRAdestroy(GENmodel **inModel)
{
    TRAmodel *mod = *(TRAmodel **) inModel;

    while (mod) {
        TRAmodel *next_mod = mod->TRAnextModel;
        TRAinstance *inst = mod->TRAinstances;
        while (inst) {
            TRAinstance *next_inst = inst->TRAnextInstance;
            FREE(inst);
            inst = next_inst;
        }
        FREE(mod);
        mod = next_mod;
    }

    *inModel = NULL;
}
