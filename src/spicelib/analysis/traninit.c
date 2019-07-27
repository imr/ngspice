/**********
Copyright 1991 Regents of the University of California.  All rights reserved.
Modified: 2000 AlansFixes
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/trandefs.h"
#include "ngspice/iferrmsg.h"
#include "ngspice/cpextern.h"

/*
 * this used to be in setup, but we need it here now
 * (must be done after mode is set as below)
 */

int TRANinit(CKTcircuit	*ckt, JOB *anal)
{
    TRANan *job = (TRANan *) anal;

    ckt->CKTfinalTime = job->TRANfinalTime;
    ckt->CKTstep      = job->TRANstep;
    ckt->CKTinitTime  = job->TRANinitTime;
    ckt->CKTmaxStep   = job->TRANmaxStep;

    /*  Maximum step size is limited to tstep given by .tran tstep tstop <tstart <tmax>>.
        May be overridden to a value (tstop - tstart)/50 by 'set nostepsizelimit'.
        Both may be overriden by setting tmax. */
    if(ckt->CKTmaxStep == 0) {
        if ((ckt->CKTstep < ( ckt->CKTfinalTime - ckt->CKTinitTime )/50.0) && !cp_getvar("nostepsizelimit", CP_BOOL, NULL, 0))
            ckt->CKTmaxStep = ckt->CKTstep;
        else
            ckt->CKTmaxStep = ( ckt->CKTfinalTime - ckt->CKTinitTime )/50.0;
    }

    ckt->CKTdelmin = 1e-11*ckt->CKTmaxStep;	/* XXX */
    ckt->CKTmode = job->TRANmode;

    return OK;
}
