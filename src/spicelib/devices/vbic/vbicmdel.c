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

#include "ngspice.h"
#include "vbicdefs.h"
#include "sperror.h"
#include "suffix.h"


int
VBICmDelete(GENmodel **inModels, IFuid modname, GENmodel *kill)
{
    VBICmodel **model = (VBICmodel**)inModels;
    VBICmodel *modfast = (VBICmodel*)kill;

    VBICmodel **oldmod;
    oldmod = model;
    for( ; *model ; model = &((*model)->VBICnextModel)) {
        if( (*model)->VBICmodName == modname || 
                (modfast && *model == modfast) ) goto delgot;
        oldmod = model;
    }
    return(E_NOMOD);

delgot:
    if( (*model)->VBICinstances ) return(E_NOTEMPTY);
    *oldmod = (*model)->VBICnextModel; /* cut deleted device out of list */
    FREE(*model);
    return(OK);

}
