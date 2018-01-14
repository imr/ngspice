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
        HFET2model *next_mod = HFET2nextModel(mod);
        HFET2instance *inst = HFET2instances(mod);
        while (inst) {
            HFET2instance *next_inst = HFET2nextInstance(inst);
            HFET2delete(GENinstanceOf(inst));
            inst = next_inst;
        }
        HFET2mDelete(GENmodelOf(mod));
        mod = next_mod;
    }

    *inModel = NULL;
}
