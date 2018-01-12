/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include "vccsdefs.h"
#include "ngspice/suffix.h"


void
VCCSdestroy(GENmodel **inModel)
{
    VCCSmodel *mod = *(VCCSmodel**) inModel;

    while (mod) {
        VCCSmodel *next_mod = VCCSnextModel(mod);
        VCCSinstance *inst = VCCSinstances(mod);
        while (inst) {
            VCCSinstance *next_inst = VCCSnextInstance(inst);
            GENinstanceFree(GENinstanceOf(inst));
            inst = next_inst;
        }
        GENmodelFree(GENmodelOf(mod));
        mod = next_mod;
    }

    *inModel = NULL;
}
