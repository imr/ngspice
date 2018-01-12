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
#include "../../../ciderlib/twod/twoddefs.h"
#include "../../../ciderlib/twod/twodext.h"
#include "ngspice/cidersupt.h"
#include "ngspice/suffix.h"


void
NUMD2destroy(GENmodel **inModel)
{
    NUMD2model *mod = *(NUMD2model **) inModel;

    while (mod) {
        NUMD2model *next_mod  = NUMD2nextModel(mod);
        NUMD2instance *inst = NUMD2instances(mod);
        while (inst) {
            NUMD2instance *next_inst = NUMD2nextInstance(inst);
            TWOdestroy(inst->NUMD2pDevice);
            GENinstanceFree(GENinstanceOf(inst));
            inst = next_inst;
        }
        GENmodelFree(GENmodelOf(mod));
        mod = next_mod;
    }

    *inModel = NULL;
}
