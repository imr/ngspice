/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

#include "ngspice/ngspice.h"
#include "ngspice/ifsim.h"
#include "ngspice/iferrmsg.h"
#include "ngspice/spardefs.h"
#include "ngspice/cktdefs.h"

#ifdef RFSPICE

/* ARGSUSED */
int
SPaskQuest(CKTcircuit *ckt, JOB *anal, int which, IFvalue *value)
{
    SPAN *job = (SPAN *) anal;

    NG_IGNORE(ckt);

    switch(which) {

    case SP_START:
        value->rValue = job->SPstartFreq;
        break;

    case SP_STOP:
        value->rValue = job->SPstopFreq ;
        break;

    case SP_STEPS:
        value->iValue = job->SPnumberSteps;
        break;

    case SP_DEC:
        if (job->SPstepType == DECADE) {
            value->iValue=1;
        } else {
            value->iValue=0;
        }
        break;

    case SP_OCT:
        if (job->SPstepType == OCTAVE) {
            value->iValue=1;
        } else {
            value->iValue=0;
        }
        break;

    case SP_LIN:
        if (job->SPstepType == LINEAR) {
            value->iValue=1;
        } else {
            value->iValue=0;
        }
        break;

    case SP_DONOISE:
        if (job->SPdoNoise)
            value->iValue = 1;
        else
            value->iValue = 0;
        break;

    default:
        return(E_BADPARM);
    }
    return(OK);
}
#endif
