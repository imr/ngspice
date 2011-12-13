/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1987 Gary W. Ng
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/ifsim.h"
#include "ngspice/iferrmsg.h"
#include "ngspice/noisedef.h"

#include "analysis.h"

int 
NsetParm(CKTcircuit *ckt, JOB *anal, int which, IFvalue *value)
{
    NOISEAN *job = (NOISEAN *) anal;

    NG_IGNORE(ckt);

    switch(which) {

    case N_OUTPUT:
	job->output = value->nValue;
	break;

    case N_OUTREF:
	job->outputRef = value->nValue;
	break;

    case N_INPUT:
	job->input = value->uValue;
	break;

    case N_DEC:
        job->NstpType = DECADE;
        break;

    case N_OCT:
        job->NstpType = OCTAVE;
        break;

    case N_LIN:
        job->NstpType = LINEAR;
        break;

    case N_STEPS:
        job->NnumSteps = value->iValue;
        break;

    case N_START:
	if (value->rValue <= 0.0) {
	    errMsg = copy("Frequency of 0 is invalid");
            job->NstartFreq = 1.0;
	    return(E_PARMVAL);
	}

        job->NstartFreq = value->rValue;
        break;

    case N_STOP:
	if (value->rValue <= 0.0) {
	    errMsg = copy("Frequency of 0 is invalid");
            job->NstartFreq = 1.0;
	    return(E_PARMVAL);
	}

        job->NstopFreq = value->rValue;
        break;

    case N_PTSPERSUM:
        job->NStpsSm = value->iValue;
        break;

    default:
        return(E_BADPARM);
    }
    return(OK);
}


static IFparm Nparms[] = {
    { "output",     N_OUTPUT,       IF_SET|IF_STRING,  "output noise summation node" },
    { "outputref",  N_OUTREF,   IF_SET|IF_STRING,  "output noise reference node" },
    { "input",      N_INPUT,        IF_SET|IF_STRING,  "input noise source" },
    { "dec",        N_DEC,          IF_SET|IF_FLAG,    "step by decades" },
    { "oct",        N_OCT,          IF_SET|IF_FLAG,    "step by octaves" },
    { "lin",        N_LIN,          IF_SET|IF_FLAG,    "step linearly" },
    { "numsteps",   N_STEPS,        IF_SET|IF_INTEGER, "number of frequencies" },
    { "start",      N_START,        IF_SET|IF_REAL,    "starting frequency" },
    { "stop",       N_STOP,         IF_SET|IF_REAL,    "ending frequency" },
    { "ptspersum",  N_PTSPERSUM,    IF_SET|IF_INTEGER, "frequency points per summary report" }
};

SPICEanalysis NOISEinfo  = {
    { 
        "NOISE",
        "Noise analysis",

        NUMELEMS(Nparms),
        Nparms
    },
    sizeof(NOISEAN),
    FREQUENCYDOMAIN,
    1,
    NsetParm,
    NaskQuest,
    NULL,
    NOISEan
};
