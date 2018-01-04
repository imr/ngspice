/**********
Copyright 1992 Regents of the University of California.  All rights reserved.
Author: 1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
**********/

/*
 * This routine deletes a NBJT instance from the circuit and frees the
 * storage it was using.
 */

#include "ngspice/ngspice.h"
#include "nbjtdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
NBJTdelete(GENmodel *inModel, IFuid name, GENinstance **kill)
{
    NBJTmodel *model = (NBJTmodel *) inModel;
    NBJTinstance **fast = (NBJTinstance **) kill;
    NBJTinstance **prev = NULL;
    NBJTinstance *here;

    for (; model; model = model->NBJTnextModel) {
        prev = &(model->NBJTinstances);
        for (here = *prev; here; here = *prev) {
            if (here->NBJTname == name || (fast && here == *fast)) {
                *prev = here->NBJTnextInstance;
                FREE(here);
                return OK;
            }
            prev = &(here->NBJTnextInstance);
        }
    }

    return E_NODEV;
}
