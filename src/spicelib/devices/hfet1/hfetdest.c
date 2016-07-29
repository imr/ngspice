/**********
Imported from MacSpice3f4 - Antony Wilson
Modified: Paolo Nenzi
**********/

#include "ngspice/ngspice.h"
#include "hfetdefs.h"
#include "ngspice/suffix.h"


void
HFETAdestroy(GENmodel **inModel)
{
    HFETAmodel *mod = *(HFETAmodel**) inModel;

    while (mod) {
        HFETAmodel *next_mod = mod->HFETAnextModel;
        HFETAinstance *inst = mod->HFETAinstances;
        while (inst) {
            HFETAinstance *next_inst = inst->HFETAnextInstance;
            FREE(inst);
            inst = next_inst;
        }
        FREE(mod);
        mod = next_mod;
    }

    *inModel = NULL;
}
