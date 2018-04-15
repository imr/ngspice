/**********
Copyright 1991 Regents of the University of California.  All rights reserved.
**********/

#include "ngspice/ngspice.h"
#include "ngspice/ifsim.h"
#include "ngspice/iferrmsg.h"
#include "ngspice/cktdefs.h"
#include "ngspice/sensdefs.h"


/* ARGSUSED */
int 
SENSask(CKTcircuit *ckt, JOB *anal, int which, IFvalue *value)
{
    SENS_AN *job = (SENS_AN *) anal;

    NG_IGNORE(ckt);

    switch (which) {

    case SENS_START:
	value->rValue = job->start_freq;
        break;

    case SENS_STOP:
	value->rValue = job->stop_freq;
        break;

    case SENS_STEPS:
	value->iValue = job->n_freq_steps;
        break;

    case SENS_DECADE:
    case SENS_OCTAVE:
    case SENS_LINEAR:
    case SENS_DC:
	value->iValue = job->step_type == which;
        break;

    case SENS_DEFTOL:
	value->rValue = job->deftol;
	break;

    case SENS_DEFPERTURB:
	value->rValue = job->defperturb;
	break;

    default:
        return(E_BADPARM);
    }
    return(OK);
}

