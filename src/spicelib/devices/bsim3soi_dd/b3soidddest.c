/**********
Copyright 1999 Regents of the University of California.  All rights reserved.
Author: 1998 Samuel Fung, Dennis Sinitsky and Stephen Tang
File: b3soidddest.c          98/5/01
Modified by Paolo Nenzi 2002
**********/

/*
 * Revision 2.1  99/9/27 Pin Su
 * BSIMDD2.1 release
 */

#include "ngspice/ngspice.h"
#include "b3soidddef.h"
#include "ngspice/suffix.h"


void
B3SOIDDdestroy(GENmodel **inModel)
{
    B3SOIDDmodel *mod = *(B3SOIDDmodel**) inModel;

    while (mod) {
        B3SOIDDmodel *next_mod = B3SOIDDnextModel(mod);
        B3SOIDDinstance *inst = B3SOIDDinstances(mod);
        while (inst) {
            B3SOIDDinstance *next_inst = inst->B3SOIDDnextInstance;
            FREE(inst);
            inst = next_inst;
        }
        FREE(mod);
        mod = next_mod;
    }

    *inModel = NULL;
}
