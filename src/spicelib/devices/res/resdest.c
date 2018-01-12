/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: Apr 2000 - Paolo Nenzi
**********/

#include "ngspice/ngspice.h"
#include "resdefs.h"


void
RESdestroy(GENmodel **inModel)
{
    RESmodel *mod = *(RESmodel **) inModel;

    while (mod) {
        RESmodel *next_mod = RESnextModel(mod);
        RESinstance *inst = RESinstances(mod);
        while (inst) {
            RESinstance *next_inst = RESnextInstance(inst);
            GENinstanceFree(GENinstanceOf(inst));
            inst = next_inst;
        }
        GENmodelFree(GENmodelOf(mod));
        mod = next_mod;
    }

    *inModel = NULL;
}
