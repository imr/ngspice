/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include "cccsdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
CCCSdelete(GENmodel *inModel, IFuid name, GENinstance **kill)
{
    CCCSmodel *model = (CCCSmodel *) inModel;
    CCCSinstance **fast = (CCCSinstance **) kill;
    CCCSinstance **prev = NULL;
    CCCSinstance *here;

    for (; model; model = model->CCCSnextModel) {
        prev = &(model->CCCSinstances);
        for (here = *prev; here; here = *prev) {
            if (here->CCCSname == name || (fast && here == *fast)) {
                *prev = here->CCCSnextInstance;
                FREE(here);
                return OK;
            }
            prev = &(here->CCCSnextInstance);
        }
    }

    return E_NODEV;
}
