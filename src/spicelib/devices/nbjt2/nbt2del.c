/**********
Copyright 1992 Regents of the University of California.  All rights reserved.
Author: 1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
**********/

/*
 * This routine deletes a NBJT2 instance from the circuit and frees the
 * storage it was using.
 */

#include "ngspice/ngspice.h"
#include "nbjt2def.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
NBJT2delete(GENmodel *inModel, IFuid name, GENinstance **kill)
{
    NBJT2model *model = (NBJT2model *) inModel;
    NBJT2instance **fast = (NBJT2instance **) kill;
    NBJT2instance **prev = NULL;
    NBJT2instance *here;

    for (; model; model = model->NBJT2nextModel) {
        prev = &(model->NBJT2instances);
        for (here = *prev; here; here = *prev) {
            if (here->NBJT2name == name || (fast && here == *fast)) {
                *prev = here->NBJT2nextInstance;
                FREE(here);
                return OK;
            }
            prev = &(here->NBJT2nextInstance);
        }
    }

    return E_NODEV;
}
