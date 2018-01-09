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
        MOS3model *next_mod = MOS3nextModel(mod);
        MOS3instance *inst = MOS3instances(mod);
        while (inst) {
            MOS3instance *next_inst = MOS3nextInstance(inst);
            FREE(inst->MOS3sens);
            FREE(inst);
            inst = next_inst;
        }
        FREE(mod);
        mod = next_mod;
    }

    *inModel = NULL;
}
