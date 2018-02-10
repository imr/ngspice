/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include "mos1defs.h"
#include "ngspice/suffix.h"


void
MOS1destroy(GENmodel **inModel)
{
    MOS1model *mod = *(MOS1model**) inModel;

    while (mod) {
        MOS1model *next_mod = mod->MOS1nextModel;
        MOS1instance *inst = mod->MOS1instances;
        while (inst) {
            MOS1instance *next_inst = inst->MOS1nextInstance;
            FREE(inst);
            inst = next_inst;
        }
        FREE(mod);
        mod = next_mod;
    }

    *inModel = NULL;
}
