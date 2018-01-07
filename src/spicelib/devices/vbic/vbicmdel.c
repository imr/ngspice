/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Model Author: 1995 Colin McAndrew Motorola
Spice3 Implementation: 2003 Dietmar Warning DAnalyse GmbH
**********/

/*
 * This routine deletes a VBIC model from the circuit and frees
 * the storage it was using.
 * returns an error if the model has instances
 */

#include "ngspice/ngspice.h"
#include "vbicdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
VBICmDelete(GENmodel **models, IFuid modname, GENmodel *kill)
{
    GENmodel **prev = models;
    GENmodel *model = *prev;

    for (; model; model = model->GENnextModel) {
        if (model->GENmodName == modname || (kill && model == kill))
            break;
        prev = &(model->GENnextModel);
    }

    if (!model)
        return E_NOMOD;

    if (model->GENinstances)
        return E_NOTEMPTY;
    *prev = model->GENnextModel;
    FREE(model);
    return OK;
}
