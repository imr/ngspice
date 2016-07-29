/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 S. Hwang
**********/

#include "ngspice/ngspice.h"
#include "mesdefs.h"
#include "ngspice/suffix.h"


void
MESdestroy(GENmodel **inModel)
{
    MESmodel *mod = *(MESmodel**) inModel;

    while (mod) {
        MESmodel *next_mod = mod->MESnextModel;
        MESinstance *inst = mod->MESinstances;
        while (inst) {
            MESinstance *next_inst = inst->MESnextInstance;
            FREE(inst);
            inst = next_inst;
        }
        FREE(mod);
        mod = next_mod;
    }

    *inModel = NULL;
}
