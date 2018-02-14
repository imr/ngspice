/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: September 2003 Paolo Nenzi
**********/

#include "ngspice/ngspice.h"
#include "capdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
CAPdelete(GENmodel *inModel, IFuid name, GENinstance **inst)
{
    CAPinstance **fast = (CAPinstance **) inst;
    CAPmodel *model = (CAPmodel *) inModel;
    CAPinstance **prev = NULL;
    CAPinstance *here;

    for (; model; model = model->CAPnextModel) {
        prev = &(model->CAPinstances);
        for (here = *prev; here; here = *prev) {
            if (here->CAPname == name || (fast && here == *fast)) {
                *prev = here->CAPnextInstance;
                FREE(here);
                return(OK);
            }
            prev = &(here->CAPnextInstance);
        }
    }

    return(E_NODEV);
}
