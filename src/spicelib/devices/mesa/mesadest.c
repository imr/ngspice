/**********
Copyright 1993: T. Ytterdal, K. Lee, M. Shur and T. A. Fjeldly. All rights reserved.
Author: Trond Ytterdal
**********/

#include "ngspice/ngspice.h"
#include "mesadefs.h"
#include "ngspice/suffix.h"


void
MESAdestroy(GENmodel **inModel)
{
    MESAmodel *mod = *(MESAmodel**) inModel;

    while (mod) {
        MESAmodel *next_mod = MESAnextModel(mod);
        MESAinstance *inst = MESAinstances(mod);
        while (inst) {
            MESAinstance *next_inst = inst->MESAnextInstance;
            FREE(inst);
            inst = next_inst;
        }
        FREE(mod);
        mod = next_mod;
    }

    *inModel = NULL;
}
