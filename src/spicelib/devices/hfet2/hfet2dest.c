/**********
Imported from MacSpice3f4 - Antony Wilson
Modified: Paolo Nenzi
**********/

#include "ngspice/ngspice.h"
#include "hfet2defs.h"
#include "ngspice/suffix.h"


void
HFET2destroy(GENmodel **inModel)
{
    HFET2model *mod = *(HFET2model**) inModel;

    while (mod) {
        HFET2model *next_mod = mod->HFET2nextModel;
        HFET2instance *inst = mod->HFET2instances;
        while (inst) {
            HFET2instance *next_inst = inst->HFET2nextInstance;
            FREE(inst);
            inst = next_inst;
        }
        FREE(mod);
        mod = next_mod;
    }

    *inModel = NULL;
}
