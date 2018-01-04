/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1987 Kanwar Jit Singh
**********/

#include "ngspice/ngspice.h"
#include "asrcdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
ASRCdelete(GENmodel *inModel, IFuid name, GENinstance **kill)
{
    ASRCinstance **fast = (ASRCinstance **) kill;
    ASRCmodel *model = (ASRCmodel *) inModel;
    ASRCinstance **prev = NULL;
    ASRCinstance *here;

    for (; model; model = model->ASRCnextModel) {
        prev = &(model->ASRCinstances);
        for (here = *prev; here; here = *prev) {
            if (here->ASRCname == name || (fast && here == *fast)) {
                *prev = here->ASRCnextInstance;
                FREE(here);
                return OK;
            }
            prev = &(here->ASRCnextInstance);
        }
    }

    return E_NODEV;
}
