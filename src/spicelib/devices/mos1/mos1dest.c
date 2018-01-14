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
        MOS1model *next_mod = MOS1nextModel(mod);
        MOS1instance *inst = MOS1instances(mod);
        while (inst) {
            MOS1instance *next_inst = MOS1nextInstance(inst);
            MOS1delete(GENinstanceOf(inst));
            inst = next_inst;
        }
        MOS1mDelete(GENmodelOf(mod));
        mod = next_mod;
    }

    *inModel = NULL;
}
