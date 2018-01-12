/**********
Copyright 1992 Regents of the University of California.  All rights reserved.
Author: 1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
**********/

/*
 * This routine deletes all NBJTs from the circuit and frees all storage they
 * were using.  The current implementation has memory leaks.
 */

#include "ngspice/ngspice.h"
#include "nbjtdefs.h"
#include "../../../ciderlib/oned/onedext.h"
#include "ngspice/cidersupt.h"
#include "ngspice/suffix.h"


void
NBJTdestroy(GENmodel **inModel)
{
    NBJTmodel *mod = *(NBJTmodel **) inModel;

    while (mod) {
        NBJTmodel *next_mod = NBJTnextModel(mod);
        NBJTinstance *inst = NBJTinstances(mod);
        while (inst) {
            NBJTinstance *next_inst = NBJTnextInstance(inst);
            ONEdestroy(inst->NBJTpDevice);
            GENinstanceFree(GENinstanceOf(inst));
            inst = next_inst;
        }
        GENmodelFree(GENmodelOf(mod));
        mod = next_mod;
    }

    *inModel = NULL;
}
