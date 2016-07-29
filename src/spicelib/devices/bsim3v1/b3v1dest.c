/**********
 * Copyright 1990 Regents of the University of California. All rights reserved.
 * File: b3v1dest.c
 * Author: 1995 Min-Chie Jeng and Mansun Chan.
 * Modified by Paolo Nenzi 2002
 **********/

/*
 * Release Notes:
 * BSIM3v3.1,   Released by yuhua  96/12/08
 */

#include "ngspice/ngspice.h"
#include "bsim3v1def.h"
#include "ngspice/suffix.h"


void
BSIM3v1destroy(GENmodel **inModel)
{
    BSIM3v1model *mod = *(BSIM3v1model**) inModel;

    while (mod) {
        BSIM3v1model *next_mod = mod->BSIM3v1nextModel;
        BSIM3v1instance *inst = mod->BSIM3v1instances;
        while (inst) {
            BSIM3v1instance *next_inst = inst->BSIM3v1nextInstance;
            FREE(inst);
            inst = next_inst;
        }
        FREE(mod);
        mod = next_mod;
    }

    *inModel = NULL;
}
