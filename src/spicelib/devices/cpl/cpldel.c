/**********
Copyright 1992 Regents of the University of California.  All rights
reserved.
Author: 1992 Charles Hough
**********/

#include "ngspice/ngspice.h"
#include "cpldefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
CPLdelete(GENmodel *inModel, IFuid name, GENinstance **inst)
{
    CPLmodel *model = (CPLmodel *) inModel;
    CPLinstance **fast = (CPLinstance **) inst;
    CPLinstance **prev = NULL;
    CPLinstance *here;

    for (; model; model = model->CPLnextModel) {
        prev = &(model->CPLinstances);
        for (here = *prev; here; here = *prev) {
            if (here->CPLname == name || (fast && here == *fast)) {
                *prev = here->CPLnextInstance;
                FREE(here);
                return OK;
            }
            prev = &(here->CPLnextInstance);
        }
    }

    return E_NODEV;
}
