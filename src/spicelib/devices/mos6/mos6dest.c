/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1989 Takayasu Sakurai
**********/

#include "ngspice/ngspice.h"
#include "mos6defs.h"
#include "ngspice/suffix.h"


void
MOS6destroy(GENmodel **inModel)
{
    MOS6model *mod = *(MOS6model**) inModel;

    while (mod) {
        MOS6model *next_mod = mod->MOS6nextModel;
        MOS6instance *inst = mod->MOS6instances;
        while (inst) {
            MOS6instance *next_inst = inst->MOS6nextInstance;
            FREE(inst);
            inst = next_inst;
        }
        FREE(mod);
        mod = next_mod;
    }

    *inModel = NULL;
}
