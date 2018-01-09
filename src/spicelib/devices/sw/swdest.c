/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Gordon Jacobs
**********/

#include "ngspice/ngspice.h"
#include "swdefs.h"
#include "ngspice/suffix.h"


void
SWdestroy(GENmodel **inModel)
{
    SWmodel *mod = *(SWmodel**) inModel;

    while (mod) {
        SWmodel *next_mod = SWnextModel(mod);
        SWinstance *inst = SWinstances(mod);
        while (inst) {
            SWinstance *next_inst = inst->SWnextInstance;
            FREE(inst);
            inst = next_inst;
        }
        FREE(mod);
        mod = next_mod;
    }

    *inModel = NULL;
}
