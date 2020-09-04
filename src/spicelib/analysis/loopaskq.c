/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1988 Jaijeet S Roychowdhury

Copyright 2020 Anamosic Ballenegger Design.  All rights reserved.
Author: 2020 Florian Ballenegger
**********/

#include "ngspice/ngspice.h"
#include "ngspice/ifsim.h"
#include "ngspice/iferrmsg.h"
#include "ngspice/cktdefs.h"
#include "ngspice/loopdefs.h"


/* ARGSUSED */
int 
LOOPaskQuest(CKTcircuit *ckt, JOB *anal, int which, IFvalue *value)
{
    LOOPAN *job = (LOOPAN *) anal;

    NG_IGNORE(ckt);

    switch(which) {

    case LOOP_START:
        value->rValue = job->LOOPstartFreq;
        break;

    case LOOP_STOP:
        value->rValue = job->LOOPstopFreq ;
        break;

    case LOOP_STEPS:
        value->iValue = job->LOOPnumSteps;
        break;

    case LOOP_DEC:
        if (job->LOOPstepType == DECADE) {
            value->iValue=1;
        } else {
            value->iValue=0;
        }
        break;

    case LOOP_OCT:
        if (job->LOOPstepType == OCTAVE) {
            value->iValue=1;
        } else {
            value->iValue=0;
        }
        break;

    case LOOP_LIN:
        if (job->LOOPstepType == LINEAR) {
            value->iValue=1;
        } else {
            value->iValue=0;
        }
        break;

    default:
        return(E_BADPARM);
    }
    return(OK);
}
