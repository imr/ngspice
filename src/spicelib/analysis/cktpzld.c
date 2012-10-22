/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
**********/

#include "ngspice/ngspice.h"
#include "ngspice/pzdefs.h"
#include "ngspice/smpdefs.h"
#include "ngspice/cktdefs.h"
#include "ngspice/complex.h"
#include "ngspice/devdefs.h"
#include "ngspice/sperror.h"


int
CKTpzLoad(CKTcircuit *ckt, SPcomplex *s)
{
    PZAN *job = (PZAN *) ckt->CKTcurJob;

    int error;
    int i;

    for (i = 0; i <= SMPmatSize(ckt->CKTmatrix); i++) {
	ckt->CKTrhs[i] = 0.0;
	ckt->CKTirhs[i] = 0.0;
    }

    SMPcClear(ckt->CKTmatrix);
    for (i = 0; i < DEVmaxnum; i++) {
        if (DEVices[i] && DEVices[i]->DEVpzLoad != NULL && ckt->CKThead[i] != NULL) {
            error = DEVices[i]->DEVpzLoad (ckt->CKThead[i], ckt, s);
            if(error) return(error);
        }
    }

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
