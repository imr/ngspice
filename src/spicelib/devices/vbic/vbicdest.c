/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Model Author: 1995 Colin McAndrew Motorola
Spice3 Implementation: 2003 Dietmar Warning DAnalyse GmbH
**********/

/*
 * This routine deletes all VBICs from the circuit and frees
 * all storage they were using.
 */

#include "ngspice/ngspice.h"
#include "vbicdefs.h"
#include "ngspice/suffix.h"


void
VBICdestroy(GENmodel **inModel)
{
    VBICmodel *mod = *(VBICmodel**) inModel;

    while (mod) {
        VBICmodel *next_mod = VBICnextModel(mod);
        VBICinstance *inst = VBICinstances(mod);
        while (inst) {
            VBICinstance *next_inst = VBICnextInstance(inst);
            FREE(inst->VBICsens);
            FREE(inst);
            inst = next_inst;
        }
        FREE(mod);
        mod = next_mod;
    }

    *inModel = NULL;
}
