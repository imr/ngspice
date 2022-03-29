/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include "ngspice/ifsim.h"
#include "ngspice/iferrmsg.h"
#include "ngspice/spardefs.h"
#include "ngspice/cktdefs.h"

#include "analysis.h"

#ifdef RFSPICE

/* ARGSUSED */
int
SPsetParm(CKTcircuit *ckt, JOB *anal, int which, IFvalue *value)
{
    SPAN *job = (SPAN *) anal;

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


static IFparm SPparms[] = {
    { "start",      SP_START,   IF_SET|IF_ASK|IF_REAL, "starting frequency" },
    { "stop",       SP_STOP,    IF_SET|IF_ASK|IF_REAL, "ending frequency" },
    { "numsteps",   SP_STEPS,IF_SET|IF_ASK|IF_INTEGER, "number of frequencies"},
    { "dec",        SP_DEC,     IF_SET|IF_FLAG, "step by decades" },
    { "oct",        SP_OCT,     IF_SET|IF_FLAG, "step by octaves" },
    { "lin",        SP_LIN,     IF_SET|IF_FLAG, "step linearly" },
    {"donoise",     SP_DONOISE, IF_SET | IF_FLAG | IF_INTEGER, "do SP noise"}
};

SPICEanalysis SPinfo  = {
    {
        "SP",
        "S-Parameters analysis",

        NUMELEMS(SPparms),
        SPparms
    },
    sizeof(SPAN),
    FREQUENCYDOMAIN,
    1,
    SPsetParm,
    SPaskQuest,
    NULL,
    SPan
};
#endif
