/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Gordon Jacobs
**********/

#include "ngspice/ngspice.h"
#include "cswdefs.h"
#include "ngspice/suffix.h"


void
CSWdestroy(GENmodel **inModel)
{
    CSWmodel *mod = *(CSWmodel**) inModel;

    while (mod) {
        CSWmodel *next_mod = CSWnextModel(mod);
        CSWinstance *inst = CSWinstances(mod);
        while (inst) {
            CSWinstance *next_inst = CSWnextInstance(inst);
            FREE(inst);
            inst = next_inst;
        }
        FREE(mod);
        mod = next_mod;
    }

    *inModel = NULL;
}
