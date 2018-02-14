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
CPLmDelete(GENmodel **inModel, IFuid modname, GENmodel *kill)
{
    CPLmodel **model = (CPLmodel **) inModel;
    CPLmodel *modfast = (CPLmodel *) kill;
    CPLinstance *here;
    CPLinstance *prev = NULL;
    CPLmodel **oldmod;

    oldmod = model;
    for (; *model; model = &((*model)->CPLnextModel)) {
        if ((*model)->CPLmodName == modname ||
            (modfast && *model == modfast)) goto delgot;
        oldmod = model;
    }

    return(E_NOMOD);

 delgot:
    *oldmod = (*model)->CPLnextModel; /* cut deleted device out of list */
    for (here = (*model)->CPLinstances; here; here = here->CPLnextInstance) {
        if (prev) FREE(prev);
        prev = here;
    }
    if (prev) FREE(prev);
    FREE(*model);
    return(OK);
}
