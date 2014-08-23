/**********
Author: Francesco Lannutti - August 2014
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
        case RELAN_AGING_STEP:
            if (value->rValue <= 0.0)
            {
                errMsg = copy ("RELAN AGING STEP is invalid, must be greater than zero.") ;
                job->RELANagingStep = 1.0 ;
                return (E_PARMVAL) ;
            }
            job->RELANagingStep = value->rValue ;
            break ;

        case RELAN_AGING_STOP:
            if (value->rValue <= 0.0)
            {
                errMsg = copy ("RELAN AGING ST0P is invalid, must be greater than zero.") ;
                job->RELANagingTotalTime = 1.0 ;
	        return (E_PARMVAL) ;
	    }
            job->RELANagingTotalTime = value->rValue ;
            break ;

        case RELAN_AGING_START:
            if (value->rValue >= job->RELANagingTotalTime)
            {
                errMsg = copy ("RELAN AGING START is invalid, must be less than RELAN AGING STOP.") ;
                job->RELANagingStartTime = 0.0 ;
                return (E_PARMVAL) ;
            }
            job->RELANagingStartTime = value->rValue ;
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
    { "relan_aging_step",  RELAN_AGING_STEP,  IF_SET|IF_REAL, "Reliability Analysis - Aging Time Step" },
    { "relan_aging_stop",  RELAN_AGING_STOP,  IF_SET|IF_REAL, "Reliability Analysis - Aging Stop Time" },
    { "relan_aging_start", RELAN_AGING_START, IF_SET|IF_REAL, "Reliability Analysis - Aging Start Time" },
    { "uic",               RELAN_UIC,         IF_SET|IF_FLAG, "Use Initial Conditions" },
} ;

SPICEanalysis RELANinfo = {
    {
        "RELAN",
        "Reliability Analysis",
        NUMELEMS (RELANparms),
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
