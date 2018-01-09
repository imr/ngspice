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
        MOS2model *next_mod = MOS2nextModel(mod);
        MOS2instance *inst = MOS2instances(mod);
        while (inst) {
            MOS2instance *next_inst = MOS2nextInstance(inst);
            FREE(inst->MOS2sens);
            FREE(inst);
            inst = next_inst;
        }
        FREE(mod);
        mod = next_mod;
    }

    *inModel = NULL;
}
