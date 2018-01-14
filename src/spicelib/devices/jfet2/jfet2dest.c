/**********
Based on jfetdest.c
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles

Modified to jfet2 for PS model definition ( Anthony E. Parker )
   Copyright 1994  Macquarie University, Sydney Australia.
**********/

#include "ngspice/ngspice.h"
#include "jfet2defs.h"
#include "ngspice/suffix.h"


void
JFET2destroy(GENmodel **inModel)
{
    JFET2model *mod = *(JFET2model**) inModel;

    while (mod) {
        JFET2model *next_mod = JFET2nextModel(mod);
        JFET2instance *inst = JFET2instances(mod);
        while (inst) {
            JFET2instance *next_inst = JFET2nextInstance(inst);
            JFET2delete(GENinstanceOf(inst));
            inst = next_inst;
        }
        JFET2mDelete(GENmodelOf(mod));
        mod = next_mod;
    }

    *inModel = NULL;
}
