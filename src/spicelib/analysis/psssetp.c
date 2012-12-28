/**********
Author: 2010-05 Stefano Perticaroli ``spertica''
**********/

#include "ngspice/ngspice.h"
#include "ngspice/ifsim.h"
#include "ngspice/iferrmsg.h"
#include "ngspice/cktdefs.h"
#include "ngspice/pssdefs.h"

#include "analysis.h"

/* ARGSUSED */
int
PSSsetParm(CKTcircuit *ckt, JOB *anal, int which, IFvalue *value)
{
    PSSan *job = (PSSan *) anal;

    NG_IGNORE(ckt);

    switch(which) {

    case GUESSED_FREQ:
        job->PSSguessedFreq = value->rValue;
        break;
    case OSC_NODE:
        job->PSSoscNode = value->nValue;
        break;
    case STAB_TIME:
        job->PSSstabTime = value->rValue;
        break;
    case PSS_POINTS:
        job->PSSpoints = value->iValue;
        break;
    case PSS_HARMS:
        job->PSSharms = value->iValue;
        break;
    case PSS_UIC:
        if(value->iValue) {
            job->PSSmode |= MODEUIC;
        }
        break;
    case SC_ITER:
        job->sc_iter = value->iValue;
        break;
    case STEADY_COEFF:
        job->steady_coeff = value->rValue;
        break;

    default:
        return(E_BADPARM);
    }
    return(OK);
}


static IFparm PSSparms[] = {
    { "fguess",    GUESSED_FREQ,	IF_SET|IF_REAL, 	"guessed frequency" },
    { "oscnode",   OSC_NODE,		IF_SET|IF_STRING,	"oscillation node" },
    { "stabtime",  STAB_TIME,		IF_SET|IF_REAL,		"stabilization time" },
    { "points",    PSS_POINTS,		IF_SET|IF_INTEGER, 	"pick equispaced number of time points in PSS" },
    { "harmonics", PSS_HARMS,		IF_SET|IF_INTEGER, 	"consider only given number of harmonics in PSS from DC" },
    { "uic",       PSS_UIC,		IF_SET|IF_INTEGER, 	"use initial conditions (1 true - 0 false)" },
    { "sc_iter",   SC_ITER,		IF_SET|IF_INTEGER, 	"maxmimum number of shooting cycle iterations" },
    { "steady_coeff",   STEADY_COEFF,	IF_SET|IF_INTEGER, 	"set steady coefficient for convergence test" }
};

SPICEanalysis PSSinfo  = {
    {
        "PSS",
        "Periodic Steady State analysis",

        sizeof(PSSparms)/sizeof(IFparm),
        PSSparms
    },
    sizeof(PSSan),
    TIMEDOMAIN,
    1,
    PSSsetParm,
    PSSaskQuest,
    PSSinit,
    DCpss
};
