/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

#include "ngspice/ngspice.h"
#include "ngspice/ifsim.h"
#include "ngspice/iferrmsg.h"
#include "ngspice/hbardefs.h"
#include "ngspice/cktdefs.h"

#ifdef WITH_HB

/* ARGSUSED */
int
HBaskQuest(CKTcircuit *ckt, JOB *anal, int which, IFvalue *value)
{
    HBAN *job = (HBAN *) anal;

    NG_IGNORE(ckt);

    switch(which) {

    case HB_F1:
        value->rValue = job->HBFreq1;
        break;

    case HB_F2:
        value->rValue = job->HBFreq2;
        break;

    default:
        return(E_BADPARM);
    }
    return(OK);
}
#endif
