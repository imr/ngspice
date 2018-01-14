/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include "inddefs.h"
#include "ngspice/suffix.h"


void
INDdestroy(GENmodel **inModel)
{
    INDmodel *mod = *(INDmodel**) inModel;

    while (mod) {
        INDmodel *next_mod = INDnextModel(mod);
        INDinstance *inst = INDinstances(mod);
        while (inst) {
            INDinstance *next_inst = INDnextInstance(inst);
            INDdelete(GENinstanceOf(inst));
            inst = next_inst;
        }
        INDmDelete(GENmodelOf(mod));
        mod = next_mod;
    }

    *inModel = NULL;
}
