/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 S. Hwang
**********/

/*
  Imported into mesa model: 2001 Paolo Nenzi
*/

#include "ngspice/ngspice.h"
#include "mesadefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
MESAmDelete(GENmodel **model, IFuid modname, GENmodel *kill)
{
    GENinstance *here;
    GENinstance *prev = NULL;
    GENmodel **oldmod;

    oldmod = model;
    for (; *model; model = &((*model)->GENnextModel)) {
        if ((*model)->GENmodName == modname || (kill && *model == kill))
            goto delgot;
        oldmod = model;
    }

    return E_NOMOD;

 delgot:
    *oldmod = (*model)->GENnextModel; /* cut deleted device out of list */
    for (here = (*model)->GENinstances; here; here = here->GENnextInstance) {
        if (prev)
            FREE(prev);
        prev = here;
    }
    if (prev)
        FREE(prev);
    FREE(*model);
    return OK;
}
