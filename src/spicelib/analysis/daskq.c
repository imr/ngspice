/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1988 Jaijeet S Roychowdhury
**********/

#include <ngspice/ngspice.h>
#include <ngspice/ifsim.h>
#include <ngspice/iferrmsg.h>
#include <ngspice/cktdefs.h>
#include <ngspice/distodef.h>


/* ARGSUSED */
int 
DaskQuest(CKTcircuit *ckt, JOB *anal, int which, IFvalue *value)
{
    NG_IGNORE(ckt);

    switch(which) {

    case D_START:
        value->rValue = ((DISTOAN*)anal)->DstartF1;
        break;

    case D_STOP:
        value->rValue = ((DISTOAN*)anal)->DstopF1 ;
        break;

    case D_STEPS:
        value->iValue = ((DISTOAN*)anal)->DnumSteps;
        break;

    case D_DEC:
        if(((DISTOAN*)anal)->DstepType == DECADE) {
            value->iValue=1;
        } else {
            value->iValue=0;
        }
        break;

    case D_OCT:
        if(((DISTOAN*)anal)->DstepType == OCTAVE) {
            value->iValue=1;
        } else {
            value->iValue=0;
        }
        break;

    case D_LIN:
        if(((DISTOAN*)anal)->DstepType == LINEAR) {
            value->iValue=1;
        } else {
            value->iValue=0;
        }
        break;

    case D_F2OVRF1:
	value->rValue = ((DISTOAN*)anal)->Df2ovrF1;
	break;
    default:
        return(E_BADPARM);
    }
    return(OK);
}
