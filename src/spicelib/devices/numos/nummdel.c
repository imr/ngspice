/**********
Copyright 1991 Regents of the University of California.  All rights reserved.
Author: 1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
**********/

/*
 * This routine deletes a NUMOS instance from the circuit and frees the
 * storage it was using.
 */

#include "ngspice/ngspice.h"
#include "numosdef.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
NUMOSdelete(GENmodel *inModel, IFuid name, GENinstance **kill)
{
    NUMOSmodel *model = (NUMOSmodel *) inModel;
    NUMOSinstance **fast = (NUMOSinstance **) kill;
    NUMOSinstance **prev = NULL;
    NUMOSinstance *here;

    for (; model; model = model->NUMOSnextModel) {
        prev = &(model->NUMOSinstances);
        for (here = *prev; here; here = *prev) {
            if (here->NUMOSname == name || (fast && here == *fast)) {
                *prev = here->NUMOSnextInstance;
                FREE(here);
                return OK;
            }
            prev = &(here->NUMOSnextInstance);
        }
    }

    return E_NODEV;
}
