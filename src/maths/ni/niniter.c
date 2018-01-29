/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1987 Gary W. Ng
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"


/*
 * NInzIter (ckt, posDrive, negDrive)
 *
 * This routine solves the adjoint system.  It assumes that the matrix has
 * already been loaded by a call to NIacIter, so it only alters the right
 * hand side vector.  The unit-valued current excitation is applied
 * between nodes posDrive and negDrive.
 */


void
NInzIter(CKTcircuit *ckt, int posDrive, int negDrive)
{
    int i;

    /* clear out the right hand side vector */

    for (i = 0; i <= SMPmatSize(ckt->CKTmatrix); i++) {
	ckt->CKTrhs [i] = 0.0;
	ckt->CKTirhs [i] = 0.0;
    }

    ckt->CKTrhs [posDrive] = 1.0;     /* apply unit current excitation */
    ckt->CKTrhs [negDrive] = -1.0;
    SMPcaSolve(ckt->CKTmatrix, ckt->CKTrhs, ckt->CKTirhs, ckt->CKTrhsSpare,
	    ckt->CKTirhsSpare);

    ckt->CKTrhs [0] = 0.0;
    ckt->CKTirhs [0] = 0.0;
}
