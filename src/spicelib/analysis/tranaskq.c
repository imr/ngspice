/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#include "ngspice.h"
#include "ifsim.h"
#include "iferrmsg.h"
#include "trandefs.h"
#include "cktdefs.h"

/* ARGSUSED */
int 
TRANaskQuest(CKTcircuit *ckt, void *anal, int which,IFvalue *value)
{
    switch(which) {

    case TRAN_TSTOP:
        value->rValue = ((TRANan *)anal)->TRANfinalTime;
        break;
    case TRAN_TSTEP:
        value->rValue = ((TRANan *)anal)->TRANstep;
        break;
    case TRAN_TSTART:
        value->rValue = ((TRANan *)anal)->TRANinitTime;
        break;
    case TRAN_TMAX:
        value->rValue = ((TRANan *)anal)->TRANmaxStep;
        break;
    case TRAN_UIC:
        if(((TRANan *)anal)->TRANmode & MODEUIC) {
            value->iValue = 1;
        } else {
            value->iValue = 0;
        }
        break;


    default:
        return(E_BADPARM);
    }
    return(OK);
}

