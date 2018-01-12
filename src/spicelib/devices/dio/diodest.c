/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include "diodefs.h"
#include "ngspice/suffix.h"


void
DIOdestroy(GENmodel **inModel)
{
    DIOmodel *mod = *(DIOmodel**) inModel;

    while (mod) {
        DIOmodel *next_mod = DIOnextModel(mod);
        DIOinstance *inst = DIOinstances(mod);
        while (inst) {
            DIOinstance *next_inst = DIOnextInstance(inst);
            GENinstanceFree(GENinstanceOf(inst));
            inst = next_inst;
        }
        GENmodelFree(GENmodelOf(mod));
        mod = next_mod;
    }

    *inModel = NULL;
}
