/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

#include "ngspice/ngspice.h"
#include "ngspice/ifsim.h"
#include "ngspice/iferrmsg.h"
#include "ngspice/acdefs.h"
#include "ngspice/cktdefs.h"


/* ARGSUSED */
int 
ACaskQuest(CKTcircuit *ckt, JOB *anal, int which, IFvalue *value)
{
    ACAN *job = (ACAN *) anal;

    NG_IGNORE(ckt);

    switch(which) {

    case AC_START:
        value->rValue = job->ACstartFreq;
        break;

    case AC_STOP:
        value->rValue = job->ACstopFreq ;
        break;

    case AC_STEPS:
        value->iValue = job->ACnumberSteps;
        break;

    case AC_DEC:
        if (job->ACstepType == DECADE) {
            value->iValue=1;
        } else {
            value->iValue=0;
        }
        break;

    case AC_OCT:
        if (job->ACstepType == OCTAVE) {
            value->iValue=1;
        } else {
            value->iValue=0;
        }
        break;

    case AC_LIN:
        if (job->ACstepType == LINEAR) {
            value->iValue=1;
        } else {
            value->iValue=0;
        }
        break;

    default:
        return(E_BADPARM);
    }
    return(OK);
}

