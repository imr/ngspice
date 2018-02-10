/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include "mos3defs.h"
#include "ngspice/suffix.h"


void
MOS3destroy(GENmodel **inModel)
{
    MOS3model *mod = *(MOS3model **) inModel;

    while (mod) {
        MOS3model *next_mod = mod->MOS3nextModel;
        MOS3instance *inst = mod->MOS3instances;
        while (inst) {
            MOS3instance *next_inst = inst->MOS3nextInstance;
            FREE(inst);
            inst = next_inst;
        }
        FREE(mod);
        mod = next_mod;
    }

    *inModel = NULL;
}
