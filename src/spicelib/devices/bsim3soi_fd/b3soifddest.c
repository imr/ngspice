/**********
Copyright 1999 Regents of the University of California.  All rights reserved.
Author: 1998 Samuel Fung, Dennis Sinitsky and Stephen Tang
Modified by Paolo Nenzi 2002
File: b3soifddest.c          98/5/01
**********/

/*
 * Revision 2.1  99/9/27 Pin Su
 * BSIMFD2.1 release
 */

#include "ngspice/ngspice.h"
#include "b3soifddef.h"
#include "ngspice/suffix.h"


void
B3SOIFDdestroy(GENmodel **inModel)
{
    B3SOIFDmodel *mod = *(B3SOIFDmodel**) inModel;

    while (mod) {
        B3SOIFDmodel *next_mod = mod->B3SOIFDnextModel;
        B3SOIFDinstance *inst = mod->B3SOIFDinstances;
        while (inst) {
            B3SOIFDinstance *next_inst = inst->B3SOIFDnextInstance;
            FREE(inst);
            inst = next_inst;
        }
        FREE(mod);
        mod = next_mod;
    }

    *inModel = NULL;
}
