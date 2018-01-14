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
        MOS6model *next_mod = MOS6nextModel(mod);
        MOS6instance *inst = MOS6instances(mod);
        while (inst) {
            MOS6instance *next_inst = MOS6nextInstance(inst);
            MOS6delete(GENinstanceOf(inst));
            inst = next_inst;
        }
        MOS6mDelete(GENmodelOf(mod));
        mod = next_mod;
    }

    *inModel = NULL;
}
