/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#include "ngspice.h"
#include "ifsim.h"
#include "iferrmsg.h"
#include "trandefs.h"
#include "cktdefs.h"

#include "analysis.h"

/* ARGSUSED */
int 
TRANsetParm(CKTcircuit *ckt, void *anal, int which, IFvalue *value)
{
    switch(which) {

    case TRAN_TSTOP:
        if (value->rValue <= 0.0) {
	        errMsg = copy("TST0P is invalid, must be greater than zero.");
                ((TRANan *)anal)->TRANfinalTime = 1.0;
	        return(E_PARMVAL);
	    }
        ((TRANan *)anal)->TRANfinalTime = value->rValue;
        break;
    case TRAN_TSTEP:
          if (value->rValue <= 0.0) {
           errMsg = copy( "TSTEP is invalid, must be greater than zero." );
           ((TRANan *)anal)->TRANstep = 1.0;
	       return(E_PARMVAL);
	    }
        ((TRANan *)anal)->TRANstep = value->rValue;
        break;
    case TRAN_TSTART:
        if (value->rValue >= ((TRANan *)anal)->TRANfinalTime ) {
	        errMsg = copy("TSTART is invalid, must be less than TSTOP.");
                ((TRANan *)anal)->TRANinitTime = 0.0;
	        return(E_PARMVAL);
	    }
        ((TRANan *)anal)->TRANinitTime = value->rValue;
        break;
    case TRAN_TMAX:
        ((TRANan *)anal)->TRANmaxStep = value->rValue;
        break;
    case TRAN_UIC:
        if(value->iValue) {
            ((TRANan *)anal)->TRANmode |= MODEUIC;
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

        sizeof(TRANparms)/sizeof(IFparm),
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
