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

#include "ngspice/ngspice.h"
#include "vbicdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
VBICdelete(GENinstance *inst)
{
    GENinstanceFree(inst);
    return OK;
}
