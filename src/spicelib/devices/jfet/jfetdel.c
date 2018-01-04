/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include "jfetdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
JFETdelete(GENmodel *inModel, IFuid name, GENinstance **kill)
{
    JFETmodel *model = (JFETmodel *) inModel;
    JFETinstance **fast = (JFETinstance **) kill;
    JFETinstance **prev = NULL;
    JFETinstance *here;

    for (; model; model = model->JFETnextModel) {
        prev = &(model->JFETinstances);
        for (here = *prev; here; here = *prev) {
            if (here->JFETname == name || (fast && here == *fast)) {
                *prev = here->JFETnextInstance;
                FREE(here);
                return OK;
            }
            prev = &(here->JFETnextInstance);
        }
    }

    return E_NODEV;
}
