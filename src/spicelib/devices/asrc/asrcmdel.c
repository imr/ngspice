/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1987 Kanwar Jit Singh
**********/

#include "ngspice/ngspice.h"
#include "asrcdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
ASRCmDelete(GENmodel **models, IFuid modname, GENmodel *kill)
{
    GENinstance *here;
    GENmodel **prev = models;
    GENmodel *model = *prev;

    for (; model; model = model->GENnextModel) {
        if (model->GENmodName == modname || (kill && model == kill))
            break;
        prev = &(model->GENnextModel);
    }

    if (!model)
        return E_NOMOD;

    *prev = model->GENnextModel;

    for (here = model->GENinstances; here;) {
        GENinstance *next_instance = here->GENnextInstance;
        FREE(((ASRCinstance*)here)->ASRCacValues);
        FREE(here);
        here = next_instance;
    }

    FREE(model);
    return OK;
}
