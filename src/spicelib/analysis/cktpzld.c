/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
**********/

#include <ngspice/ngspice.h>
#include <ngspice/pzdefs.h>
#include <ngspice/smpdefs.h>
#include <ngspice/cktdefs.h>
#include <ngspice/complex.h>
#include <ngspice/devdefs.h>
#include <ngspice/sperror.h>


int
CKTpzLoad(CKTcircuit *ckt, SPcomplex *s)
{
    PZAN *job = (PZAN *) (ckt->CKTcurJob);
    int error;
    int i;
#ifdef PARALLEL_ARCH
    long type = MT_PZLOAD, length = 1;
#endif /* PARALLEL_ARCH */

    for (i = 0; i <= SMPmatSize(ckt->CKTmatrix); i++) {
	ckt->CKTrhs[i] = 0.0;
	ckt->CKTirhs[i] = 0.0;
    }

    SMPcClear(ckt->CKTmatrix);
    for (i = 0; i < DEVmaxnum; i++) {
        if (DEVices[i] && DEVices[i]->DEVpzLoad != NULL && ckt->CKThead[i] != NULL) {
            error = DEVices[i]->DEVpzLoad (ckt->CKThead[i], ckt, s);
#ifdef PARALLEL_ARCH
	    if (error) goto combine;
#else
            if(error) return(error);
#endif /* PARALLEL_ARCH */
        }
    }
#ifdef PARALLEL_ARCH
combine:
    /* See if any of the DEVload functions bailed. If not, proceed. */
    IGOP_( &type, &error, &length, "max" );
    if (error == OK) {
	SMPcCombine(ckt->CKTmatrix, ckt->CKTrhs, ckt->CKTrhsSpare,
		    ckt->CKTirhs, ckt->CKTirhsSpare );
    } else {
	return(error);
    }
#endif /* PARALLEL_ARCH */

    if (job->PZbalance_col && job->PZsolution_col) {
	SMPcAddCol(ckt->CKTmatrix, job->PZbalance_col, job->PZsolution_col);
	/* AC sources ?? XXX */
    }

    if (job->PZsolution_col) {
	SMPcZeroCol(ckt->CKTmatrix, job->PZsolution_col);
    }

    /* Driving function (current source) */
    if (job->PZdrive_pptr)
	*job->PZdrive_pptr = 1.0;
    if (job->PZdrive_nptr)
	*job->PZdrive_nptr = -1.0;

    return(OK);
}
