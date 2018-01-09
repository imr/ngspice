/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1987 Kanwar Jit Singh
**********/

#include "ngspice/ngspice.h"
#include "asrcdefs.h"
#include "ngspice/suffix.h"
#include "ngspice/inpdefs.h"


void
ASRCdestroy(GENmodel **inModel)
{
    ASRCmodel *mod = *(ASRCmodel**) inModel;

    while (mod) {
        ASRCmodel *next_mod = ASRCnextModel(mod);
        ASRCinstance *inst = ASRCinstances(mod);
        while (inst) {
            ASRCinstance *next_inst = inst->ASRCnextInstance;
            INPfreeTree(inst->ASRCtree);
            FREE(inst->ASRCacValues);
            FREE(inst->ASRCposPtr);
            FREE(inst->ASRCvars);
            FREE(inst);
            inst = next_inst;
        }
        FREE(mod);
        mod = next_mod;
    }

    FREE(asrc_vals);
    FREE(asrc_derivs);
    asrc_nvals = 0;

    *inModel = NULL;
}
