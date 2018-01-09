/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1995 Min-Chie Jeng and Mansun Chan.
File: b3v0dest.c
**********/

#include "ngspice/ngspice.h"
#include "bsim3v0def.h"
#include "ngspice/suffix.h"


void
BSIM3v0destroy(GENmodel **inModel)
{
    BSIM3v0model *mod = *(BSIM3v0model**) inModel;

    while (mod) {
        BSIM3v0model *next_mod = BSIM3v0nextModel(mod);
        BSIM3v0instance *inst = BSIM3v0instances(mod);
        while (inst) {
            BSIM3v0instance *next_inst = inst->BSIM3v0nextInstance;
            FREE(inst);
            inst = next_inst;
        }
        FREE(mod);
        mod = next_mod;
    }

    *inModel = NULL;
}
