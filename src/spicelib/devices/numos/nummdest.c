/**********
Copyright 1991 Regents of the University of California.  All rights reserved.
Author: 1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
**********/

/*
 * This routine deletes all NUMOSs from the circuit and frees all storage
 * they were using.  The current implementation has memory leaks.
 */

#include "ngspice/ngspice.h"
#include "numosdef.h"
#include "../../../ciderlib/twod/twoddefs.h"
#include "../../../ciderlib/twod/twodext.h"
#include "ngspice/suffix.h"


void
NUMOSdestroy(GENmodel **inModel)
{
    NUMOSmodel *mod = *(NUMOSmodel **) inModel;

    while (mod) {
        NUMOSmodel *next_mod = NUMOSnextModel(mod);
        NUMOSinstance *inst = NUMOSinstances(mod);
        while (inst) {
            NUMOSinstance *next_inst = NUMOSnextInstance(inst);
            TWOdestroy(inst->NUMOSpDevice);
            FREE(inst);
            inst = next_inst;
        }
        FREE(mod);
        mod = next_mod;
    }

    *inModel = NULL;
}
