/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1987 Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include "urcdefs.h"
#include "ngspice/suffix.h"


void
URCdestroy(GENmodel **inModel)
{
    URCmodel *mod = *(URCmodel **) inModel;

    while (mod) {
        URCmodel *next_mod = URCnextModel(mod);
        URCinstance *inst = URCinstances(mod);
        while (inst) {
            URCinstance *next_inst = inst->URCnextInstance;
            FREE(inst);
            inst = next_inst;
        }
        FREE(mod);
        mod = next_mod;
    }

    *inModel = NULL;
}
