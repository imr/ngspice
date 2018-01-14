/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1987 Kanwar Jit Singh
**********/

#include "ngspice/ngspice.h"
#include "asrcdefs.h"
#include "ngspice/suffix.h"


void
ASRCdestroy(GENmodel **inModel)
{
    ASRCmodel *mod = *(ASRCmodel**) inModel;

    while (mod) {
        ASRCmodel *next_mod = ASRCnextModel(mod);
        ASRCinstance *inst = ASRCinstances(mod);
        while (inst) {
            ASRCinstance *next_inst = ASRCnextInstance(inst);
            ASRCdelete(GENinstanceOf(inst));
            inst = next_inst;
        }
        ASRCmDelete(GENmodelOf(mod));
        mod = next_mod;
    }

    FREE(asrc_vals);
    FREE(asrc_derivs);
    asrc_nvals = 0;

    *inModel = NULL;
}
