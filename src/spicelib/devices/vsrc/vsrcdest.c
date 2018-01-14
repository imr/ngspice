/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include "vsrcdefs.h"
#include "ngspice/suffix.h"


void
VSRCdestroy(GENmodel **inModel)
{
    VSRCmodel *mod = *(VSRCmodel**) inModel;

    while (mod) {
        VSRCmodel *next_mod = VSRCnextModel(mod);
        VSRCinstance *inst = VSRCinstances(mod);
        while (inst) {
            VSRCinstance *next_inst = VSRCnextInstance(inst);
            VSRCdelete(GENinstanceOf(inst));
            inst = next_inst;
        }
        VSRCmDelete(GENmodelOf(mod));
        mod = next_mod;
    }

    *inModel = NULL;
}
