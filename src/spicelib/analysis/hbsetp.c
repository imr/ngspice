/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include "ngspice/ifsim.h"
#include "ngspice/iferrmsg.h"
#include "ngspice/hbardefs.h"
#include "ngspice/cktdefs.h"

#include "analysis.h"

#ifdef RFSPICE

/* ARGSUSED */
int
HBsetParm(CKTcircuit *ckt, JOB *anal, int which, IFvalue *value)
{
    HBAN *job = (HBAN *) anal;

    NG_IGNORE(ckt);

    switch(which) {

    case SP_START:
	if (value->rValue < 0.0) {
	    errMsg = copy("Frequency of < 0 is invalid for AC start");
            job->SPstartFreq = 1.0;
	    return(E_PARMVAL);
	}

        job->SPstartFreq = value->rValue;
        break;

    case SP_STOP:
	if (value->rValue < 0.0) {
	    errMsg = copy("Frequency of < 0 is invalid for AC stop");
            job->SPstartFreq = 1.0;
	    return(E_PARMVAL);
	}

        job->SPstopFreq = value->rValue;
        break;

    case SP_STEPS:
        job->SPnumberSteps = value->iValue;
        break;

    case SP_DEC:
        if(value->iValue) {
            job->SPstepType = DECADE;
        } else {
            if (job->SPstepType == DECADE) {
                job->SPstepType = 0;
            }
        }
        break;

    case SP_OCT:
        if(value->iValue) {
                job->SPstepType = OCTAVE;
        } else {
            if (job->SPstepType == OCTAVE) {
                job->SPstepType = 0;
            }
        }
        break;

    case SP_LIN:
        if(value->iValue) {
            job->SPstepType = LINEAR;
        } else {
            if (job->SPstepType == LINEAR) {
                job->SPstepType = 0;
            }
        }
        break;

    case SP_DONOISE:
        job->SPdoNoise = value->iValue == 1;
        break;

    default:
        return(E_BADPARM);
    }
    return(OK);
}


static IFparm HBparms[] = {
    { "f1",      SP_START,   IF_SET|IF_ASK|IF_REAL, "fundamental frequency" },
    { "f2",       SP_STOP,    IF_SET|IF_ASK|IF_REAL, "second frequency" }
};

SPICEanalysis HBinfo  = {
    {
        "HB",
        "Harmonic Balance analysis",

        NUMELEMS(HBparms),
        HBparms
    },
    sizeof(HBAN),
    FREQUENCYDOMAIN,
    1,
    HBsetParm,
    HBaskQuest,
    NULL,
    HBan
};
#endif
