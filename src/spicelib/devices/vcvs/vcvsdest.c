/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include "vcvsdefs.h"
#include "ngspice/suffix.h"


void
VCVSdestroy(GENmodel **inModel)
{
    VCVSmodel *mod = *(VCVSmodel **) inModel;

    while (mod) {
        VCVSmodel *next_mod = VCVSnextModel(mod);
        VCVSinstance *inst = VCVSinstances(mod);
        while (inst) {
            VCVSinstance *next_inst = VCVSnextInstance(inst);
            GENinstanceFree(GENinstanceOf(inst));
            inst = next_inst;
        }
        GENmodelFree(GENmodelOf(mod));
        mod = next_mod;
    }

    *inModel = NULL;
}
