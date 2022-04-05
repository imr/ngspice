/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/devdefs.h"
#include "ngspice/ifsim.h"
#include "ngspice/sperror.h"



GENmodel *
CKTfndMod(CKTcircuit *ckt, IFuid modname)
{
    return ckt ? nghash_find(ckt->MODnameHash, modname) : NULL;
}
