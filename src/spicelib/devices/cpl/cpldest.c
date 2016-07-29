/**********
Copyright 1992 Regents of the University of California.  All rights
reserved.
Author: 1992 Charles Hough
**********/

#include "ngspice/ngspice.h"
#include "cpldefs.h"
#include "ngspice/suffix.h"


void
CPLdestroy(GENmodel **inModel)
{
    CPLmodel *mod = *(CPLmodel **) inModel;

    while (mod) {
        CPLmodel *next_mod = mod->CPLnextModel;
        CPLinstance *inst = mod->CPLinstances;
        while (inst) {
            CPLinstance *next_inst = inst->CPLnextInstance;
            FREE(inst);
            inst = next_inst;
        }
        FREE(mod);
        mod = next_mod;
    }

    *inModel = NULL;
}
