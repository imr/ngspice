/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Hong J. Park, Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include "bsim2def.h"
#include "ngspice/sperror.h"
#include "ngspice/gendefs.h"
#include "ngspice/suffix.h"


int
B2delete(GENmodel *inModel, IFuid name, GENinstance **kill)
{
    B2instance **fast = (B2instance **) kill;
    B2model *model = (B2model *) inModel;
    B2instance **prev = NULL;
    B2instance *here;

    for (; model; model = model->B2nextModel) {
        prev = &(model->B2instances);
        for (here = *prev; here; here = *prev) {
            if (here->B2name == name || (fast && here == *fast)) {
                *prev = here->B2nextInstance;
                FREE(here);
                return OK;
            }
            prev = &(here->B2nextInstance);
        }
    }

    return E_NODEV;
}
