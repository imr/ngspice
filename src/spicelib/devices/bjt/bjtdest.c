/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

/*
 * This routine deletes all BJTs from the circuit and frees
 * all storage they were using.
 */

#include "ngspice/ngspice.h"
#include "bjtdefs.h"
#include "ngspice/suffix.h"


void
BJTdestroy(GENmodel **inModel)
{
    BJTmodel *mod = *(BJTmodel**) inModel;

    while (mod) {
        BJTmodel *next_mod = mod->BJTnextModel;
        BJTinstance *inst = mod->BJTinstances;
        while (inst) {
            BJTinstance *next_inst = inst->BJTnextInstance;
            FREE(inst);
            inst = next_inst;
        }
        FREE(mod);
        mod = next_mod;
    }

    *inModel = NULL;
}
