/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1988 Jaijeet S Roychowdhury
**********/

#include "ngspice/ngspice.h"
#include "ngspice/ifsim.h"
#include "ngspice/iferrmsg.h"
#include "ngspice/cktdefs.h"
#include "ngspice/distodef.h"

#include "analysis.h"

/* ARGSUSED */
int 
DsetParm(CKTcircuit *ckt, JOB *anal, int which, IFvalue *value)
{
    DISTOAN *job = (DISTOAN *) anal;

    NG_IGNORE(ckt);

    switch(which) {

    case D_START:
	if (value->rValue <= 0.0) {
	    errMsg = copy("Frequency of 0 is invalid");
            job->DstartF1 = 1.0;
	    return(E_PARMVAL);
	}

        job->DstartF1 = value->rValue;
        break;

    case D_STOP:
	if (value->rValue <= 0.0) {
	    errMsg = copy("Frequency of 0 is invalid");
            job->DstartF1 = 1.0;
	    return(E_PARMVAL);
	}

        job->DstopF1 = value->rValue;
        break;

    case D_STEPS:
        job->DnumSteps = value->iValue;
        break;

    case D_DEC:
        job->DstepType = DECADE;
        break;

    case D_OCT:
        job->DstepType = OCTAVE;
        break;

    case D_LIN:
        job->DstepType = LINEAR;
        break;

    case D_F2OVRF1:
        job->Df2ovrF1 = value->rValue;
        job->Df2wanted = 1;
        break;

    default:
        return(E_BADPARM);
    }
    return(OK);
}


static IFparm Dparms[] = {
    { "start",      D_START,   IF_SET|IF_REAL, "starting frequency" },
    { "stop",       D_STOP,    IF_SET|IF_REAL, "ending frequency" },
    { "numsteps",   D_STEPS,   IF_SET|IF_INTEGER,  "number of frequencies" },
    { "dec",        D_DEC,     IF_SET|IF_FLAG, "step by decades" },
    { "oct",        D_OCT,     IF_SET|IF_FLAG, "step by octaves" },
    { "lin",        D_LIN,     IF_SET|IF_FLAG, "step linearly" },
    { "f2overf1",   D_F2OVRF1, IF_SET|IF_REAL, "ratio of F2 to F1" },
};

SPICEanalysis DISTOinfo  = {
    { 
        "DISTO",
        "Small signal distortion analysis",

        NUMELEMS(Dparms),
        Dparms
    },
    sizeof(DISTOAN),
    FREQUENCYDOMAIN,
    1,
    DsetParm,
    DaskQuest,
    NULL,
    DISTOan
};
