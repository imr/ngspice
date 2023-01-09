/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

/* CKTaccept(ckt)
 *
 * this is a driver program to iterate through all the various accept
 * functions provided for the circuit elements in the given circuit */

#include <string.h>

#include "ngspice/config.h"
#include "ngspice/devdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/ifsim.h"
#include "ngspice/devdefs.h"

#include "dev.h"

int
CKTaccept(CKTcircuit *ckt)
{
    int i;
    int error;
    SPICEdev **devs;

#ifdef PREDICTOR
    double *temp;
    int size;
#endif

    devs = devices();
    for (i = 0; i < DEVmaxnum; i++) {
        if ( devs[i] && devs[i]->DEVaccept && ckt->CKThead[i] ) {
            error = devs[i]->DEVaccept (ckt, ckt->CKThead[i]);
            if (error)
		return(error);
        }
    }
#ifdef PREDICTOR
    /* now, move the sols vectors around */
    temp = ckt->CKTsols[7];
    for ( i=7;i>0;i--) {
        ckt->CKTsols[i] = ckt->CKTsols[i-1];
    }
    ckt->CKTsols[0]=temp;
    size = SMPmatSize(ckt->CKTmatrix);
    memcpy(ckt->CKTsols[0], ckt->CKTrhs, (size + 1)*sizeof(double));
#endif /* PREDICTOR */
    return(OK);
}
