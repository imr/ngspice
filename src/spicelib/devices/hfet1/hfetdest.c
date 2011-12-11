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
    HFETAmodel **model = (HFETAmodel**)inModel;
    HFETAinstance *here;
    HFETAinstance *prev = NULL;
    HFETAmodel *mod = *model;
    HFETAmodel *oldmod = NULL;

    for( ; mod ; mod = mod->HFETAnextModel) {
        if(oldmod) FREE(oldmod);
        oldmod = mod;
        prev = NULL;
        for(here = mod->HFETAinstances ; here ; here = here->HFETAnextInstance) {
            if(prev) FREE(prev);
            prev = here;
        }
        if(prev) FREE(prev);
    }
    if(oldmod) FREE(oldmod);
    *model = NULL;
    return;
}
