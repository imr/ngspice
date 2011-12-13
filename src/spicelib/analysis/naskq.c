/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1987 Gary W. Ng
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/ifsim.h"
#include "ngspice/iferrmsg.h"
#include "ngspice/noisedef.h"


int 
NaskQuest(CKTcircuit *ckt, JOB *anal, int which, IFvalue *value)
{
    NOISEAN *job = (NOISEAN *) anal;

    NG_IGNORE(ckt);

    switch(which) {

    case N_OUTPUT:
        value->nValue = job->output;
        break;

    case N_OUTREF:
        value->nValue = job->outputRef;
        break;

    case N_INPUT:
        value->uValue = job->input;
        break;

    case N_DEC:
        if (job->NstpType == DECADE) {
            value->iValue=1;
        } else {
            value->iValue=0;
        }
        break;

    case N_OCT:
        if (job->NstpType == OCTAVE) {
            value->iValue=1;
        } else {
            value->iValue=0;
        }
        break;

    case N_LIN:
        if (job->NstpType == LINEAR) {
            value->iValue=1;
        } else {
            value->iValue=0;
        }
        break;

    case N_STEPS:
        value->iValue = job->NnumSteps;
        break;

    case N_START:
        value->rValue = job->NstartFreq;
        break;

    case N_STOP:
        value->rValue = job->NstopFreq;
        break;

    case N_PTSPERSUM:
        value->iValue = job->NStpsSm;
	break;

    default:
        return(E_BADPARM);
    }
    return(OK);
}

