/**********
Author: Francesco Lannutti - July 2015
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
        case RELAN_TSTART:
            value->rValue = job->RELANinitTime ;
            break ;
        case RELAN_TSTOP:
            value->rValue = job->RELANfinalTime ;
            break ;
        case RELAN_TSTEP:
            value->rValue = job->RELANstep ;
            break ;
        case RELAN_TMAX:
            value->rValue = job->RELANmaxStep ;
            break ;
        case RELAN_UIC:
            if (job->RELANmode & MODEUIC) {
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
