/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Gordon Jacobs
**********/

#include "ngspice/ngspice.h"
#include "cswdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
CSWdelete(GENmodel *inModel, IFuid name, GENinstance **kill)
{
    CSWmodel *model = (CSWmodel *) inModel;
    CSWinstance **fast = (CSWinstance **) kill;
    CSWinstance **prev = NULL;
    CSWinstance *here;

    for (; model; model = model->CSWnextModel) {
        prev = &(model->CSWinstances);
        for (here = *prev; here; here = *prev) {
            if (here->CSWname == name || (fast && here == *fast)) {
                *prev = here->CSWnextInstance;
                FREE(here);
                return OK;
            }
            prev = &(here->CSWnextInstance);
        }
    }

    return E_NODEV;
}
