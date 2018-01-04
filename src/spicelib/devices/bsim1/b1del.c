/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Hong J. Park, Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include "bsim1def.h"
#include "ngspice/sperror.h"
#include "ngspice/gendefs.h"
#include "ngspice/suffix.h"


int
B1delete(GENmodel *inModel, IFuid name, GENinstance **kill)
{
    B1instance **fast = (B1instance **) kill;
    B1model *model = (B1model *) inModel;
    B1instance **prev = NULL;
    B1instance *here;

    for (; model; model = model->B1nextModel) {
        prev = &(model->B1instances);
        for (here = *prev; here; here = *prev) {
            if (here->B1name == name || (fast && here == *fast)) {
                *prev = here->B1nextInstance;
                FREE(here);
                return OK;
            }
            prev = &(here->B1nextInstance);
        }
    }

    return E_NODEV;
}
