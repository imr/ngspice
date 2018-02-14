/**********
Copyright 1991 Regents of the University of California.  All rights reserved.
Author: 1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
**********/

/*
 * This routine deletes a NUMOS model from the circuit and frees the storage
 * it was using. returns an error if the model has instances
 */

#include "ngspice/ngspice.h"
#include "numosdef.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
NUMOSmDelete(GENmodel **inModel, IFuid modname, GENmodel *kill)
{
    NUMOSmodel **model = (NUMOSmodel **) inModel;
    NUMOSmodel *modfast = (NUMOSmodel *) kill;
    NUMOSmodel **oldmod;

    oldmod = model;
    for (; *model; model = &((*model)->NUMOSnextModel)) {
        if ((*model)->NUMOSmodName == modname ||
            (modfast && *model == modfast))
            goto delgot;
        oldmod = model;
    }

    return(E_NOMOD);

 delgot:
    if ((*model)->NUMOSinstances)
        return(E_NOTEMPTY);
    *oldmod = (*model)->NUMOSnextModel;   /* cut deleted device out of list */
    FREE(*model);
    return(OK);
}
