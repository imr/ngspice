/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Hong J. Park, Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include "bsim2def.h"
#include "ngspice/suffix.h"


void
B2destroy(GENmodel **inModel)
{
    B2model *mod = *(B2model**) inModel;

    while (mod) {
        B2model *next_mod = B2nextModel(mod);
        B2instance *inst = B2instances(mod);
        while (inst) {
            B2instance *next_inst = B2nextInstance(inst);
            B2delete(GENinstanceOf(inst));
            inst = next_inst;
        }
        B2mDelete(GENmodelOf(mod));
        mod = next_mod;
    }

    *inModel = NULL;
}
