/**********
Copyright 1991 Regents of the University of California.  All rights reserved.
**********/

#include "ngspice.h"
#include "ifsim.h"
#include "iferrmsg.h"
#include "cktdefs.h"
#include "sensdefs.h"

#include "analysis.h"

/* ARGSUSED */
int 
SENSsetParam(CKTcircuit *ckt, void *anal, int which, IFvalue *value)
{
    SENS_AN	*sinfo = (SENS_AN *) anal;

    switch(which) {

    case SENS_POS:
        sinfo->output_pos = (CKTnode *) value->nValue;
        sinfo->output_neg = NULL;
        sinfo->output_volt = 1;
        sinfo->step_type = SENS_DC;
        break;

    case SENS_NEG:
        sinfo->output_neg = (CKTnode *) value->nValue;
        break;

    case SENS_SRC:
        sinfo->output_src = value->uValue;
        sinfo->output_volt = 0;
        sinfo->step_type = SENS_DC;
        break;

    case SENS_NAME:
        sinfo->output_name = value->sValue;
        break;

    case SENS_START:
        sinfo->start_freq = value->rValue;
        break;

    case SENS_STOP:
        sinfo->stop_freq = value->rValue;
        break;

    case SENS_STEPS:
        sinfo->n_freq_steps = value->iValue;
        break;

    case SENS_DEC:
	sinfo->step_type = SENS_DECADE;
        break;

    case SENS_OCT:
	sinfo->step_type = SENS_OCTAVE;
        break;

    case SENS_LIN:
	sinfo->step_type = SENS_LINEAR;
	break;

    case SENS_DC:
	sinfo->step_type = SENS_DC;
	break;

    case SENS_DEFTOL:
	sinfo->deftol = value->rValue;
	break;

    case SENS_DEFPERTURB:
	sinfo->defperturb = value->rValue;
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
    { "dec",        SENS_DEC,     IF_SET|IF_FLAG, "step by decades" },
    { "oct",        SENS_OCT,     IF_SET|IF_FLAG, "step by octaves" },
    { "lin",        SENS_LIN,     IF_SET|IF_FLAG, "step linearly" },
    { "dc",         SENS_DC,      IF_SET|IF_FLAG, "analysis at DC" },
};

SPICEanalysis SENSinfo  = {
    { 
        "SENS",
        "Sensitivity analysis",
        sizeof(SENSparms)/sizeof(IFparm),
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
