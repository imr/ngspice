/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: September 2003 Paolo Nenzi
**********/

#include "ngspice/ngspice.h"
#include "capdefs.h"
#include "ngspice/suffix.h"


void
CAPdestroy(GENmodel **inModel)
{
    CAPmodel *mod = *(CAPmodel**) inModel;

    while (mod) {
        CAPmodel *next_mod = CAPnextModel(mod);
        CAPinstance *inst = CAPinstances(mod);
        while (inst) {
            CAPinstance *next_inst = CAPnextInstance(inst);
            GENinstanceFree(GENinstanceOf(inst));
            inst = next_inst;
        }
        GENmodelFree(GENmodelOf(mod));
        mod = next_mod;
    }

    *inModel = NULL;
}
