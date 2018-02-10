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
        VSRCmodel *next_mod = mod->VSRCnextModel;
        VSRCinstance *inst = mod->VSRCinstances;
        while (inst) {
            VSRCinstance *next_inst = inst->VSRCnextInstance;
            FREE(inst);
            inst = next_inst;
        }
        FREE(mod);
        mod = next_mod;
    }

    *inModel = NULL;
}
