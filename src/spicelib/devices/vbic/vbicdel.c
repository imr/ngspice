/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Model Author: 1995 Colin McAndrew Motorola
Spice3 Implementation: 2003 Dietmar Warning DAnalyse GmbH
**********/

/*
 * This routine deletes a VBIC instance from the circuit and frees
 * the storage it was using.
 */

#include "ngspice.h"
#include "vbicdefs.h"
#include "sperror.h"
#include "suffix.h"


int
VBICdelete(GENmodel *inModel, IFuid name, GENinstance **kill)
{
    VBICmodel *model = (VBICmodel*)inModel;
    VBICinstance **fast = (VBICinstance**)kill;

    VBICinstance **prev = NULL;
    VBICinstance *here;

    for( ; model ; model = model->VBICnextModel) {
        prev = &(model->VBICinstances);
        for(here = *prev; here ; here = *prev) {
            if(here->VBICname == name || (fast && here==*fast) ) {
                *prev= here->VBICnextInstance;
                FREE(here);
                return(OK);
            }
            prev = &(here->VBICnextInstance);
        }
    }
    return(E_NODEV);
}
