/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1988 Jaijeet S Roychowdhury
**********/

#include "ngspice.h"
#include "ifsim.h"
#include "iferrmsg.h"
#include "cktdefs.h"
#include "distodef.h"

#include "analysis.h"

/* ARGSUSED */
int 
DsetParm(CKTcircuit *ckt, void *anal, int which, IFvalue *value)
{
    switch(which) {

    case D_START:
	if (value->rValue <= 0.0) {
	    errMsg = copy("Frequency of 0 is invalid");
            ((DISTOAN*)anal)->DstartF1 = 1.0;
	    return(E_PARMVAL);
	}

        ((DISTOAN*)anal)->DstartF1 = value->rValue;
        break;

    case D_STOP:
	if (value->rValue <= 0.0) {
	    errMsg = copy("Frequency of 0 is invalid");
            ((DISTOAN*)anal)->DstartF1 = 1.0;
	    return(E_PARMVAL);
	}

        ((DISTOAN*)anal)->DstopF1 = value->rValue;
        break;

    case D_STEPS:
        ((DISTOAN*)anal)->DnumSteps = value->iValue;
        break;

    case D_DEC:
        ((DISTOAN*)anal)->DstepType = DECADE;
        break;

    case D_OCT:
        ((DISTOAN*)anal)->DstepType = OCTAVE;
        break;

    case D_LIN:
        ((DISTOAN*)anal)->DstepType = LINEAR;
        break;

    case D_F2OVRF1:
        ((DISTOAN*)anal)->Df2ovrF1 = value->rValue;
	((DISTOAN*)anal)->Df2wanted = 1;
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

        sizeof(Dparms)/sizeof(IFparm),
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
