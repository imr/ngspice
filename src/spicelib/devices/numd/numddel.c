/**********
Copyright 1992 Regents of the University of California.  All rights reserved.
Author: 1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
**********/

#include "ngspice/ngspice.h"
#include "numddefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
NUMDdelete(GENmodel *inModel, IFuid name, GENinstance **kill)
{
    NUMDmodel *model = (NUMDmodel *) inModel;
    NUMDinstance **fast = (NUMDinstance **) kill;
    NUMDinstance **prev = NULL;
    NUMDinstance *here;

    for (; model; model = model->NUMDnextModel) {
        prev = &(model->NUMDinstances);
        for (here = *prev; here; here = *prev) {
            if (here->NUMDname == name || (fast && here == *fast)) {
                *prev = here->NUMDnextInstance;
                FREE(here);
                return OK;
            }
            prev = &(here->NUMDnextInstance);
        }
    }

    return E_NODEV;
}
