/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#include "ngspice/config.h"
#include "ngspice/cktdefs.h"
#include "ngspice/devdefs.h"
#include "ngspice/sperror.h"
#include "string.h"


GENinstance *
CKTfndDev(CKTcircuit *ckt, IFuid name)
{
    return ckt ? nghash_find(ckt->DEVnameHash, name) : NULL;
}
