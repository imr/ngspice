/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include "inddefs.h"
#include "ngspice/suffix.h"


void
MUTdestroy(GENmodel **inModel)
{
    MUTmodel *mod = *(MUTmodel**) inModel;

    while (mod) {
        MUTmodel *next_mod = mod->MUTnextModel;
        MUTinstance *inst = mod->MUTinstances;
        while (inst) {
            MUTinstance *next_inst = inst->MUTnextInstance;
            FREE(inst);
            inst = next_inst;
        }
        FREE(mod);
        mod = next_mod;
    }

    *inModel = NULL;
}
