/**********
Copyright 1992 Regents of the University of California.  All rights reserved.
Author: 1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
**********/

#include "ngspice/ngspice.h"
#include "numddefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
NUMDmDelete(GENmodel **models, IFuid modname, GENmodel *kill)
{
    GENinstance *here;
    GENmodel **prev = models;
    GENmodel *model = *prev;

    for (; model; model = model->GENnextModel) {
        if (model->GENmodName == modname || (kill && model == kill))
            goto delgot;
        prev = &(model->GENnextModel);
    }

    return E_NOMOD;

 delgot:
    *prev = model->GENnextModel;
    for (here = model->GENinstances; here;) {
        GENinstance *next_instance = here->GENnextInstance;
        FREE(here);
        here = next_instance;
    }
    FREE(model);
    return OK;
}
