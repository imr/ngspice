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
        HFETAmodel *next_mod = HFETAnextModel(mod);
        HFETAinstance *inst = HFETAinstances(mod);
        while (inst) {
            HFETAinstance *next_inst = HFETAnextInstance(inst);
            GENinstanceFree(GENinstanceOf(inst));
            inst = next_inst;
        }
        GENmodelFree(GENmodelOf(mod));
        mod = next_mod;
    }

    *inModel = NULL;
}
