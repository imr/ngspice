/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#include "ngspice.h"
#include "ifsim.h"
#include "iferrmsg.h"
#include "opdefs.h"
#include "cktdefs.h"


/* ARGSUSED */
int 
DCOaskQuest(CKTcircuit *ckt, void *anal, int which, IFvalue *value)
{
    return(E_BADPARM);
}

