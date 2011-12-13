/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1988 Jaijeet S Roychowdhury
**********/

#include "ngspice/ngspice.h"
#include "ngspice/ifsim.h"
#include "ngspice/iferrmsg.h"
#include "ngspice/cktdefs.h"
#include "ngspice/distodef.h"


/* ARGSUSED */
int 
DaskQuest(CKTcircuit *ckt, JOB *anal, int which, IFvalue *value)
{
    DISTOAN *job = (DISTOAN *) anal;

    NG_IGNORE(ckt);

    switch(which) {

    case D_START:
        value->rValue = job->DstartF1;
        break;

    case D_STOP:
        value->rValue = job->DstopF1 ;
        break;

    case D_STEPS:
        value->iValue = job->DnumSteps;
        break;

    case D_DEC:
        if (job->DstepType == DECADE) {
            value->iValue=1;
        } else {
            value->iValue=0;
        }
        break;

    case D_OCT:
        if (job->DstepType == OCTAVE) {
            value->iValue=1;
        } else {
            value->iValue=0;
        }
        break;

    case D_LIN:
        if (job->DstepType == LINEAR) {
            value->iValue=1;
        } else {
            value->iValue=0;
        }
        break;

    case D_F2OVRF1:
	value->rValue = job->Df2ovrF1;
	break;
    default:
        return(E_BADPARM);
    }
    return(OK);
}
