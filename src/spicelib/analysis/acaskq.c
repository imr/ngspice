/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

#include "ngspice.h"
#include "ifsim.h"
#include "iferrmsg.h"
#include "acdefs.h"
#include "cktdefs.h"


/* ARGSUSED */
int 
ACaskQuest(CKTcircuit *ckt, void *anal, int which, IFvalue *value)
{
    switch(which) {

    case AC_START:
        value->rValue = ((ACAN*)anal)->ACstartFreq;
        break;

    case AC_STOP:
        value->rValue = ((ACAN*)anal)->ACstopFreq ;
        break;

    case AC_STEPS:
        value->iValue = ((ACAN*)anal)->ACnumberSteps;
        break;

    case AC_DEC:
        if(((ACAN*)anal)->ACstepType == DECADE) {
            value->iValue=1;
        } else {
            value->iValue=0;
        }
        break;

    case AC_OCT:
        if(((ACAN*)anal)->ACstepType == OCTAVE) {
            value->iValue=1;
        } else {
            value->iValue=0;
        }
        break;

    case AC_LIN:
        if(((ACAN*)anal)->ACstepType == LINEAR) {
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

