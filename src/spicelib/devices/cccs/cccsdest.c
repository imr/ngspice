/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include "cccsdefs.h"
#include "ngspice/suffix.h"


void
CCCSdestroy(GENmodel **inModel)
{
    CCCSmodel *mod = *(CCCSmodel**) inModel;

    while (mod) {
        CCCSmodel *next_mod = CCCSnextModel(mod);
        CCCSinstance *inst = CCCSinstances(mod);
        while (inst) {
            CCCSinstance *next_inst = CCCSnextInstance(inst);
            GENinstanceFree(GENinstanceOf(inst));
            inst = next_inst;
        }
        GENmodelFree(GENmodelOf(mod));
        mod = next_mod;
    }

    *inModel = NULL;
}
