/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1987 Gary W. Ng
**********/

#include "ngspice.h"
#include "cktdefs.h"
#include "ifsim.h"
#include "iferrmsg.h"
#include "noisedef.h"


int 
NaskQuest(CKTcircuit *ckt, void *anal, int which, IFvalue *value)
{
    switch(which) {

    case N_OUTPUT:
        value->nValue = ((NOISEAN*)anal)->output;
        break;

    case N_OUTREF:
        value->nValue = ((NOISEAN*)anal)->outputRef;
        break;

    case N_INPUT:
        value->uValue = ((NOISEAN*)anal)->input;
        break;

    case N_DEC:
        if(((NOISEAN*)anal)->NstpType == DECADE) {
            value->iValue=1;
        } else {
            value->iValue=0;
        }
        break;

    case N_OCT:
        if(((NOISEAN*)anal)->NstpType == OCTAVE) {
            value->iValue=1;
        } else {
            value->iValue=0;
        }
        break;

    case N_LIN:
        if(((NOISEAN*)anal)->NstpType == LINEAR) {
            value->iValue=1;
        } else {
            value->iValue=0;
        }
        break;

    case N_STEPS:
        value->iValue = ((NOISEAN*)anal)->NnumSteps;
        break;

    case N_START:
        value->rValue = ((NOISEAN*)anal)->NstartFreq;
        break;

    case N_STOP:
        value->rValue = ((NOISEAN*)anal)->NstopFreq;
        break;

    case N_PTSPERSUM:
        value->iValue = ((NOISEAN*)anal)->NStpsSm;
	break;

    default:
        return(E_BADPARM);
    }
    return(OK);
}

