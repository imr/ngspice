/**********
Author: Francesco Lannutti - August 2014
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/relandefs.h"
#include "ngspice/iferrmsg.h"

int
RELANinit (CKTcircuit *ckt, JOB *anal)
{
    RELANan *job = (RELANan *) anal ;

    ckt->CKTagingTotalTime = job->RELANagingTotalTime ;
    ckt->CKTagingStartTime = job->RELANagingStartTime ;
    ckt->CKTagingStep      = job->RELANagingStep ;

    ckt->CKTmode = job->RELANmode ;

    return (OK) ;
}
