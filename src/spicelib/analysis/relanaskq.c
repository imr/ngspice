/**********
Author: Francesco Lannutti - August 2014
**********/

#include "ngspice/ngspice.h"
#include "ngspice/ifsim.h"
#include "ngspice/iferrmsg.h"
#include "ngspice/relandefs.h"
#include "ngspice/cktdefs.h"

int 
RELANaskQuest (CKTcircuit *ckt, JOB *anal, int which, IFvalue *value)
{
    RELANan *job = (RELANan *) anal ;

    NG_IGNORE (ckt) ;

    switch (which)
    {
        case RELAN_AGING_STEP:
            value->rValue = job->RELANagingStep ;
            break ;

        case RELAN_AGING_STOP:
            value->rValue = job->RELANagingTotalTime ;
            break ;

        case RELAN_AGING_START:
            value->rValue = job->RELANagingStartTime ;
            break ;

        case RELAN_UIC:
            if (job->RELANmode & MODEUIC)
            {
                value->iValue = 1 ;
            } else {
                value->iValue = 0 ;
            }
            break ;

        default:
            return (E_BADPARM) ;
    }

    return (OK) ;
}
