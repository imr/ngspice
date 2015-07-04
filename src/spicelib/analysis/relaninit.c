/**********
Author: Francesco Lannutti - July 2015
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/relandefs.h"
#include "ngspice/iferrmsg.h"

int
RELANinit (CKTcircuit *ckt, JOB *anal)
{
    RELANan *job = (RELANan *) anal ;

    ckt->CKTinitTime  = job->RELANinitTime ;
    ckt->CKTfinalTime = job->RELANfinalTime ;
    ckt->CKTstep      = job->RELANstep ;
    ckt->CKTmaxStep   = job->RELANmaxStep ;

    if (ckt->CKTmaxStep == 0)
    {
        if (ckt->CKTstep < (ckt->CKTfinalTime - ckt->CKTinitTime) / 50.0)
        {
            ckt->CKTmaxStep = ckt->CKTstep ;
        } else {
            ckt->CKTmaxStep = (ckt->CKTfinalTime - ckt->CKTinitTime) / 50.0 ;
        }
    }

    ckt->CKTdelmin = 1e-11 * ckt->CKTmaxStep ;
    ckt->CKTmode = job->RELANmode ;

    return (OK) ;
}
