/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include "mos2defs.h"
#include "ngspice/suffix.h"


void
MOS2destroy(GENmodel **inModel)
{
    MOS2model *mod = *(MOS2model **) inModel;

    while (mod) {
        MOS2model *next_mod = mod->MOS2nextModel;
        MOS2instance *inst = mod->MOS2instances;
        while (inst) {
            MOS2instance *next_inst = inst->MOS2nextInstance;
            FREE(inst);
            inst = next_inst;
        }
        FREE(mod);
        mod = next_mod;
    }

    *inModel = NULL;
}
