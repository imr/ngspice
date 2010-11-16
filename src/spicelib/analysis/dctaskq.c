/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#include "ngspice.h"
#include "ifsim.h"
#include "iferrmsg.h"
#include "trcvdefs.h"
#include "cktdefs.h"


/* ARGSUSED */
int 
DCTaskQuest(CKTcircuit *ckt, JOB *anal, int which, IFvalue *value)
{
    NG_IGNORE(ckt);
    NG_IGNORE(anal);
    NG_IGNORE(value);
    switch(which) {

    default:
	break;
    }
    /* NOTREACHED */ /* TEMPORARY until cases get added */
    return(E_BADPARM);
}

