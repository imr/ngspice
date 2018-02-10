/**********
Copyright 1992 Regents of the University of California.  All rights reserved.
Author: 1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
**********/

/*
 * This routine deletes all NUMD2s from the circuit and frees all storage
 * they were using.  The current implementation has memory leaks.
 */

#include "ngspice/ngspice.h"
#include "numd2def.h"
#include "ngspice/suffix.h"


void
NUMD2destroy(GENmodel **inModel)
{
    NUMD2model *mod = *(NUMD2model **) inModel;

    while (mod) {
        NUMD2model *next_mod  = mod->NUMD2nextModel;
        NUMD2instance *inst = mod->NUMD2instances;
        while (inst) {
            NUMD2instance *next_inst = inst->NUMD2nextInstance;
            FREE(inst);
            inst = next_inst;
        }
        FREE(mod);
        mod = next_mod;
    }

    *inModel = NULL;
}
