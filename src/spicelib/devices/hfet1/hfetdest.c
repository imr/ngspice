
#include "ngspice.h"
#include <stdio.h>
#include "hfetdefs.h"
#include "suffix.h"


void
HFETAdestroy(inModel)
GENmodel **inModel;
{
    HFETAmodel **model = (HFETAmodel**)inModel;
    HFETAinstance *here;
    HFETAinstance *prev = NULL;
    HFETAmodel *mod = *model;
    HFETAmodel *oldmod = NULL;

    for( ; mod ; mod = mod->HFETAnextModel) {
        if(oldmod) FREE(oldmod);
        oldmod = mod;
        prev = (HFETAinstance *)NULL;
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
