/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#include "ngspice.h"
#include "ifsim.h"
#include "iferrmsg.h"
#include "acdefs.h"
#include "cktdefs.h"

#include "analysis.h"

/* ARGSUSED */
int 
ACsetParm(CKTcircuit *ckt, void *anal, int which, IFvalue *value)
{
    switch(which) {

    case AC_START:
	if (value->rValue <= 0.0) {
	    errMsg = copy("Frequency of 0 is invalid for AC start");
            ((ACAN*)anal)->ACstartFreq = 1.0;
	    return(E_PARMVAL);
	}

        ((ACAN*)anal)->ACstartFreq = value->rValue;
        break;

    case AC_STOP:
	if (value->rValue <= 0.0) {
	    errMsg = copy("Frequency of 0 is invalid for AC stop");
            ((ACAN*)anal)->ACstartFreq = 1.0;
	    return(E_PARMVAL);
	}

        ((ACAN*)anal)->ACstopFreq = value->rValue;
        break;

    case AC_STEPS:
        ((ACAN*)anal)->ACnumberSteps = value->iValue;
        break;

    case AC_DEC:
        if(value->iValue) {
            ((ACAN*)anal)->ACstepType = DECADE;
        } else {
            if( ((ACAN*)anal)->ACstepType == DECADE) {
                ((ACAN*)anal)->ACstepType = 0;
            }
        }
        break;

    case AC_OCT:
        if(value->iValue) {
                ((ACAN*)anal)->ACstepType = OCTAVE;
        } else {
            if( ((ACAN*)anal)->ACstepType == OCTAVE) {
                ((ACAN*)anal)->ACstepType = 0;
            }
        }
        break;

    case AC_LIN:
        if(value->iValue) {
            ((ACAN*)anal)->ACstepType = LINEAR;
        } else {
            if( ((ACAN*)anal)->ACstepType == LINEAR) {
                ((ACAN*)anal)->ACstepType = 0;
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

        sizeof(ACparms)/sizeof(IFparm),
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
