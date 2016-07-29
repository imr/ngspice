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
        JFET2model *next_mod = mod->JFET2nextModel;
        JFET2instance *inst = mod->JFET2instances;
        while (inst) {
            JFET2instance *next_inst = inst->JFET2nextInstance;
            FREE(inst);
            inst = next_inst;
        }
        FREE(mod);
        mod = next_mod;
    }

    *inModel = NULL;
}
