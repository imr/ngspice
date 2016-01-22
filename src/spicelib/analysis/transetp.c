/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include "ngspice/ifsim.h"
#include "ngspice/iferrmsg.h"
#include "ngspice/trandefs.h"
#include "ngspice/cktdefs.h"

#include "analysis.h"

/* ARGSUSED */
int 
TRANsetParm(CKTcircuit *ckt, JOB *anal, int which, IFvalue *value)
{
    TRANan *job = (TRANan *) anal;

    NG_IGNORE(ckt);

    switch(which) {

    case TRAN_TSTOP:
        if (value->rValue <= 0.0) {
	        errMsg = copy("TSTOP is invalid, must be greater than zero.");
                job->TRANfinalTime = 1.0;
	        return(E_PARMVAL);
	    }
        job->TRANfinalTime = value->rValue;
        break;
    case TRAN_TSTEP:
          if (value->rValue <= 0.0) {
           errMsg = copy( "TSTEP is invalid, must be greater than zero." );
           job->TRANstep = 1.0;
	       return(E_PARMVAL);
	    }
        job->TRANstep = value->rValue;
        break;
    case TRAN_TSTART:
        if (value->rValue >= job->TRANfinalTime) {
	        errMsg = copy("TSTART is invalid, must be less than TSTOP.");
                job->TRANinitTime = 0.0;
	        return(E_PARMVAL);
	    }
        job->TRANinitTime = value->rValue;
        break;
    case TRAN_TMAX:
        job->TRANmaxStep = value->rValue;
        break;
    case TRAN_UIC:
        if(value->iValue) {
            job->TRANmode |= MODEUIC;
        }
        break;

    default:
        return(E_BADPARM);
    }
    return(OK);
}


static IFparm TRANparms[] = {
    { "tstart",     TRAN_TSTART,    IF_SET|IF_REAL, "starting time" },
    { "tstop",      TRAN_TSTOP,     IF_SET|IF_REAL, "ending time" },
    { "tstep",      TRAN_TSTEP,     IF_SET|IF_REAL, "time step" },
    { "tmax",       TRAN_TMAX,      IF_SET|IF_REAL, "maximum time step" },
    { "uic",        TRAN_UIC,       IF_SET|IF_FLAG, "use initial conditions" },
};

SPICEanalysis TRANinfo  = {
    { 
        "TRAN",
        "Transient analysis",

        NUMELEMS(TRANparms),
        TRANparms
    },
    sizeof(TRANan),
    TIMEDOMAIN,
    1,
    TRANsetParm,
    TRANaskQuest,
    TRANinit,
    DCtran
};
