/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

/* CKTaccept(ckt)
 * this is a driver program to iterate through all the various
 * accept functions provided for the circuit elements in the
 * given circuit 
 */

#include "ngspice.h"
#include <stdio.h>
#include "smpdefs.h"
#include "cktdefs.h"
#include "devdefs.h"
#include "sperror.h"


int
CKTaccept(CKTcircuit *ckt)
{
    extern SPICEdev *DEVices[];

    int i;
    int error;

    for (i=0;i<DEVmaxnum;i++) {
        if ( ((*DEVices[i]).DEVaccept != NULL) && (ckt->CKThead[i] != NULL) ){
            error = (*((*DEVices[i]).DEVaccept))(ckt,ckt->CKThead[i]);
            if(error) return(error);
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
    for(i=0;i<=size;i++) {
        ckt->CKTsols[0][i]=ckt->CKTrhs[i];
    }
#endif /* PREDICTOR */
    return(OK);
}
