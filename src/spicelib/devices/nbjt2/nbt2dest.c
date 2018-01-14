/**********
Copyright 1992 Regents of the University of California.  All rights reserved.
Author: 1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
**********/

/*
 * This routine deletes all NBJT2s from the circuit and frees all storage
 * they were using.  The current implementation has memory leaks.
 */

#include "ngspice/ngspice.h"
#include "nbjt2def.h"
#include "ngspice/suffix.h"


void
NBJT2destroy(GENmodel **inModel)
{
    NBJT2model *mod = *(NBJT2model **) inModel;

    while (mod) {
        NBJT2model *next_mod = NBJT2nextModel(mod);
        NBJT2instance *inst = NBJT2instances(mod);
        while (inst) {
            NBJT2instance *next_inst = NBJT2nextInstance(inst);
            NBJT2delete(GENinstanceOf(inst));
            inst = next_inst;
        }
        NBJT2mDelete(GENmodelOf(mod));
        mod = next_mod;
    }

    *inModel = NULL;
}
