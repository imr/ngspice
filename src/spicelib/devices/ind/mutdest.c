/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include "inddefs.h"
#include "ngspice/suffix.h"


#ifdef MUTUAL

void
MUTdestroy(GENmodel **inModel)
{
    MUTmodel *mod = *(MUTmodel**) inModel;

    while (mod) {
        MUTmodel *next_mod = MUTnextModel(mod);
        MUTinstance *inst = MUTinstances(mod);
        while (inst) {
            MUTinstance *next_inst = MUTnextInstance(inst);
            FREE(inst);
            inst = next_inst;
        }
        FREE(mod);
        mod = next_mod;
    }

    *inModel = NULL;
}

#endif
