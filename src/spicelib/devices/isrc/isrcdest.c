/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include "isrcdefs.h"
#include "ngspice/suffix.h"


void
ISRCdestroy(GENmodel **inModel)
{
    ISRCmodel *mod = *(ISRCmodel**) inModel;

    while (mod) {
        ISRCmodel *next_mod = mod->ISRCnextModel;
        ISRCinstance *inst = mod->ISRCinstances;
        while (inst) {
            ISRCinstance *next_inst = inst->ISRCnextInstance;
            FREE(inst->ISRCcoeffs);
            FREE(inst);
            inst = next_inst;
        }
        FREE(mod);
        mod = next_mod;
    }

    *inModel = NULL;
}
