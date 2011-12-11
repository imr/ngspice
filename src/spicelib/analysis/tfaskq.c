/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include "ngspice/ifsim.h"
#include "ngspice/iferrmsg.h"
#include "ngspice/trcvdefs.h"
#include "ngspice/cktdefs.h"


/* ARGSUSED */
int 
TFaskQuest(CKTcircuit *ckt, JOB *anal, int which, IFvalue *value)
{
    NG_IGNORE(value);
    NG_IGNORE(anal);
    NG_IGNORE(ckt);

    switch(which) {

    default:
	break;
    }
    return(E_BADPARM);
}

