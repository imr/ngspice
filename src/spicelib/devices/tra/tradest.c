/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include "tradefs.h"
#include "ngspice/suffix.h"


void
TRAdestroy(GENmodel **inModel)
{
    TRAmodel *mod = *(TRAmodel **) inModel;

    while (mod) {
        TRAmodel *next_mod = TRAnextModel(mod);
        TRAinstance *inst = TRAinstances(mod);
        while (inst) {
            TRAinstance *next_inst = TRAnextInstance(inst);
            TRAdelete(GENinstanceOf(inst));
            inst = next_inst;
        }
        TRAmDelete(GENmodelOf(mod));
        mod = next_mod;
    }

    *inModel = NULL;
}
