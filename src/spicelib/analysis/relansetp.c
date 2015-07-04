/**********
Author: Francesco Lannutti - July 2015
**********/

#include "ngspice/ngspice.h"
#include "ngspice/ifsim.h"
#include "ngspice/iferrmsg.h"
#include "ngspice/relandefs.h"
#include "ngspice/cktdefs.h"

#include "analysis.h"

int 
RELANsetParm (CKTcircuit *ckt, JOB *anal, int which, IFvalue *value)
{
    RELANan *job = (RELANan *) anal ;

    NG_IGNORE (ckt) ;

    switch (which)
    {
        case RELAN_TSTART:
            if (value->rValue >= job->RELANfinalTime)
            {
                errMsg = copy ("TSTART is invalid, must be less than TSTOP.") ;
                job->RELANinitTime = 0.0 ;
                return (E_PARMVAL) ;
            }
            job->RELANinitTime = value->rValue ;
            break ;
        case RELAN_TSTOP:
            if (value->rValue <= 0.0)
            {
                errMsg = copy ("TST0P is invalid, must be greater than zero.") ;
                job->RELANfinalTime = 1.0 ;
                return (E_PARMVAL) ;
            }
            job->RELANfinalTime = value->rValue ;
            break ;
        case RELAN_TSTEP:
            if (value->rValue <= 0.0)
            {
                errMsg = copy ("TSTEP is invalid, must be greater than zero.") ;
                job->RELANstep = 1.0 ;
                return (E_PARMVAL) ;
	    }
            job->RELANstep = value->rValue ;
            break ;
        case RELAN_TMAX:
            job->RELANmaxStep = value->rValue ;
            break ;
        case RELAN_UIC:
            if (value->iValue)
            {
                job->RELANmode |= MODEUIC ;
            }
            break ;
        default:
            return (E_BADPARM) ;
    }

    return (OK) ;
}


static IFparm RELANparms [] = {
    { "relan_aging_start",     RELAN_TSTART,    IF_SET|IF_REAL, "starting time" },
    { "relan_aging_stop",      RELAN_TSTOP,     IF_SET|IF_REAL, "ending time" },
    { "relan_aging_step",      RELAN_TSTEP,     IF_SET|IF_REAL, "time step" },
    { "relan_aging_max",       RELAN_TMAX,      IF_SET|IF_REAL, "maximum time step" },
    { "uic",                   RELAN_UIC,       IF_SET|IF_FLAG, "use initial conditions" },
} ;

SPICEanalysis RELANinfo = {
    { 
        "RELAN",
        "Reliability analysis",

        NUMELEMS(RELANparms),
        RELANparms
    },
    sizeof(RELANan),
    TIMEDOMAIN,
    1,
    RELANsetParm,
    RELANaskQuest,
    RELANinit,
    RELANanalysis
} ;
