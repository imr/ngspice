/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Hong J. Park, Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include "bsim1def.h"
#include "ngspice/suffix.h"


void
B1destroy(GENmodel **inModel)
{
    B1model *mod = *(B1model**) inModel;

    while (mod) {
        B1model *next_mod = B1nextModel(mod);
        B1instance *inst = B1instances(mod);
        while (inst) {
            B1instance *next_inst = inst->B1nextInstance;
            FREE(inst);
            inst = next_inst;
        }
        FREE(mod);
        mod = next_mod;
    }

    *inModel = NULL;
}
