/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#include <ngspice/ngspice.h>
#include <ngspice/ifsim.h>
#include <ngspice/iferrmsg.h>
#include <ngspice/trandefs.h>
#include <ngspice/cktdefs.h>

/* ARGSUSED */
int 
TRANaskQuest(CKTcircuit *ckt, JOB *anal, int which,IFvalue *value)
{
    NG_IGNORE(ckt);

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

