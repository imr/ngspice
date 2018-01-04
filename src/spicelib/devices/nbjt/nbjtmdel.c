/**********
Copyright 1992 Regents of the University of California.  All rights reserved.
Author: 1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
**********/

/*
 * This routine deletes a NBJT model from the circuit and frees the storage
 * it was using. returns an error if the model has instances
 */

#include "ngspice/ngspice.h"
#include "nbjtdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
NBJTmDelete(GENmodel **model, IFuid modname, GENmodel *kill)
{
    GENmodel **oldmod;

    oldmod = model;
    for (; *model; model = &((*model)->GENnextModel)) {
        if ((*model)->GENmodName == modname || (kill && *model == kill))
            goto delgot;
        oldmod = model;
    }

    return E_NOMOD;

 delgot:
    if ((*model)->GENinstances)
        return E_NOTEMPTY;
    *oldmod = (*model)->GENnextModel;    /* cut deleted device out of list */
    FREE(*model);
    return OK;
}
