/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: Alan Gillespie
**********/

#include "ngspice/ngspice.h"
#include "mos9defs.h"
#include "ngspice/suffix.h"


void
MOS9destroy(GENmodel **inModel)
{
    MOS9model *mod = *(MOS9model **) inModel;

    while (mod) {
        MOS9model *next_mod = mod->MOS9nextModel;
        MOS9instance *inst = mod->MOS9instances;
        while (inst) {
            MOS9instance *next_inst = inst->MOS9nextInstance;
            FREE(inst);
            inst = next_inst;
        }
        FREE(mod);
        mod = next_mod;
    }

    *inModel = NULL;
}
