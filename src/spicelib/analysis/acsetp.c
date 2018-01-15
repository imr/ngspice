/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include "ngspice/ifsim.h"
#include "ngspice/iferrmsg.h"
#include "ngspice/acdefs.h"
#include "ngspice/cktdefs.h"

#include "analysis.h"

/* ARGSUSED */
int 
ACsetParm(CKTcircuit *ckt, JOB *anal, int which, IFvalue *value)
{
    ACAN *job = (ACAN *) anal;

    NG_IGNORE(ckt);

    switch(which) {

    case AC_START:
	if (value->rValue < 0.0) {
	    errMsg = copy("Frequency of < 0 is invalid for AC start");
            job->ACstartFreq = 1.0;
	    return(E_PARMVAL);
	}

        job->ACstartFreq = value->rValue;
        break;

    case AC_STOP:
	if (value->rValue < 0.0) {
	    errMsg = copy("Frequency of < 0 is invalid for AC stop");
            job->ACstartFreq = 1.0;
	    return(E_PARMVAL);
	}

        job->ACstopFreq = value->rValue;
        break;

    case AC_STEPS:
        job->ACnumberSteps = value->iValue;
        break;

    case AC_DEC:
        if(value->iValue) {
            job->ACstepType = DECADE;
        } else {
            if (job->ACstepType == DECADE) {
                job->ACstepType = 0;
            }
        }
        break;

    case AC_OCT:
        if(value->iValue) {
                job->ACstepType = OCTAVE;
        } else {
            if (job->ACstepType == OCTAVE) {
                job->ACstepType = 0;
            }
        }
        break;

    case AC_LIN:
        if(value->iValue) {
            job->ACstepType = LINEAR;
        } else {
            if (job->ACstepType == LINEAR) {
                job->ACstepType = 0;
            }
        }
        break;

    default:
        return(E_BADPARM);
    }
    return(OK);
}


static IFparm ACparms[] = {
    { "start",      AC_START,   IF_SET|IF_ASK|IF_REAL, "starting frequency" },
    { "stop",       AC_STOP,    IF_SET|IF_ASK|IF_REAL, "ending frequency" },
    { "numsteps",   AC_STEPS,IF_SET|IF_ASK|IF_INTEGER, "number of frequencies"},
    { "dec",        AC_DEC,     IF_SET|IF_FLAG, "step by decades" },
    { "oct",        AC_OCT,     IF_SET|IF_FLAG, "step by octaves" },
    { "lin",        AC_LIN,     IF_SET|IF_FLAG, "step linearly" }
};

SPICEanalysis ACinfo  = {
    { 
        "AC",
        "A.C. Small signal analysis",

        NUMELEMS(ACparms),
        ACparms
    },
    sizeof(ACAN),
    FREQUENCYDOMAIN,
    1,
    ACsetParm,
    ACaskQuest,
    NULL,
    ACan
};
