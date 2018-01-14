/**********
Copyright 1992 Regents of the University of California.  All rights reserved.
Author: 1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
**********/

/*
 * This routine deletes all NUMDs from the circuit and frees all storage they
 * were using.  The current implementation has memory leaks.
 */

#include "ngspice/ngspice.h"
#include "numddefs.h"
#include "ngspice/cidersupt.h"
#include "ngspice/suffix.h"


void
NUMDdestroy(GENmodel **inModel)
{
    NUMDmodel *mod = *(NUMDmodel **) inModel;

    while (mod) {
        NUMDmodel *next_mod = NUMDnextModel(mod);
        NUMDinstance *inst = NUMDinstances(mod);
        while (inst) {
            NUMDinstance *next_inst = NUMDnextInstance(inst);
            NUMDdelete(GENinstanceOf(inst));
            inst = next_inst;
        }
        NUMDmDelete(GENmodelOf(mod));
        mod = next_mod;
    }

    *inModel = NULL;
}
