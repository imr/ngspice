/**********
Copyright 1991 Regents of the University of California.  All rights reserved.
**********/

#include "ngspice/ngspice.h"
#include "ngspice/ifsim.h"
#include "ngspice/iferrmsg.h"
#include "ngspice/cktdefs.h"
#include "ngspice/sensdefs.h"

#include "analysis.h"

/* ARGSUSED */
int 
SENSsetParam(CKTcircuit *ckt, JOB *anal, int which, IFvalue *value)
{
    SENS_AN *job = (SENS_AN *) anal;

    NG_IGNORE(ckt);

    switch(which) {

    case SENS_POS:
        job->output_pos = value->nValue;
        job->output_neg = NULL;
        job->output_volt = 1;
        job->step_type = SENS_DC;
        break;

    case SENS_NEG:
        job->output_neg = value->nValue;
        break;

    case SENS_SRC:
        job->output_src = value->uValue;
        job->output_volt = 0;
        job->step_type = SENS_DC;
        break;

    case SENS_NAME:
        job->output_name = value->sValue;
        break;

    case SENS_START:
        job->start_freq = value->rValue;
        break;

    case SENS_STOP:
        job->stop_freq = value->rValue;
        break;

    case SENS_STEPS:
        job->n_freq_steps = value->iValue;
        break;

    case SENS_DECADE:
	job->step_type = SENS_DECADE;
        break;

    case SENS_OCTAVE:
	job->step_type = SENS_OCTAVE;
        break;

    case SENS_LINEAR:
	job->step_type = SENS_LINEAR;
	break;

    case SENS_DC:
	job->step_type = SENS_DC;
	break;

    case SENS_DEFTOL:
	job->deftol = value->rValue;
	break;

    case SENS_DEFPERTURB:
	job->defperturb = value->rValue;
	break;

    default:
        return(E_BADPARM);
    }
    return(OK);
}


static IFparm SENSparms[] = {
    /* TF like parameters */
    { "outpos",     SENS_POS, IF_SET|IF_ASK|IF_NODE, "output positive node" },
    { "outneg",     SENS_NEG, IF_SET|IF_ASK|IF_NODE, "output negative node" },
    { "outsrc",     SENS_SRC, IF_SET|IF_ASK|IF_INSTANCE, "output current" },
    { "outname",    SENS_NAME, IF_SET|IF_ASK|IF_STRING,
	    "Name of output variable" },

    /* AC parameters */
    { "start",      SENS_START, IF_SET|IF_ASK|IF_REAL, "starting frequency" },
    { "stop",       SENS_STOP,  IF_SET|IF_ASK|IF_REAL, "ending frequency" },
    { "numsteps",   SENS_STEPS,IF_SET|IF_ASK|IF_INTEGER,
	    "number of frequencies"},
    { "dec",        SENS_DECADE,  IF_SET|IF_FLAG, "step by decades" },
    { "oct",        SENS_OCTAVE,  IF_SET|IF_FLAG, "step by octaves" },
    { "lin",        SENS_LINEAR,  IF_SET|IF_FLAG, "step linearly" },
    { "dc",         SENS_DC,      IF_SET|IF_FLAG, "analysis at DC" },
};

SPICEanalysis SENSinfo  = {
    { 
        "SENS",
        "Sensitivity analysis",
        NUMELEMS(SENSparms),
        SENSparms
    },
    sizeof(SENS_AN),
    FREQUENCYDOMAIN,
    1,
    SENSsetParam,
    SENSask,
    NULL,
    sens_sens
};
