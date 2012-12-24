/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1988 Jaijeet S. Roychowdhury
**********/

    /*
     * NIdIter(ckt)
     *
     * This is absolutely the same as NIacIter, except that the RHS
     * vector is stored before acLoad so that it is not lost. Moreover,
     * acLoading is done only if reordering is necessary
     *
     */

#include "ngspice/ngspice.h"
#include "ngspice/trandefs.h"
#include "ngspice/cktdefs.h"
#include "ngspice/sperror.h"


int
NIdIter(CKTcircuit *ckt)
{
    int error;
    int ignore;
    double *temp;

    ckt->CKTnoncon = 0;
    goto skip;
retry:
    ckt->CKTnoncon=0;

    temp = ckt->CKTrhs;
    ckt->CKTrhs = ckt->CKTrhsSpare;
    ckt->CKTrhsSpare = temp;

    temp = ckt->CKTirhs;
    ckt->CKTirhs = ckt->CKTirhsSpare;
    ckt->CKTirhsSpare = temp;

    error = CKTacLoad(ckt);
    if(error) return(error);
    
    temp = ckt->CKTrhs;
    ckt->CKTrhs = ckt->CKTrhsSpare;
    ckt->CKTrhsSpare = temp;

    temp = ckt->CKTirhs;
    ckt->CKTirhs = ckt->CKTirhsSpare;
    ckt->CKTirhsSpare = temp;

skip:
    if(ckt->CKTniState & NIACSHOULDREORDER) {
        error = SMPcReorder(ckt->CKTmatrix,ckt->CKTpivotAbsTol,
                ckt->CKTpivotRelTol,&ignore);
        ckt->CKTniState &= ~NIACSHOULDREORDER;
        if(error != 0) {
            /* either singular equations or no memory, in either case,
             * let caller handle problem
             */
            return(error);
        }
    } else {
        error = SMPcLUfac(ckt->CKTmatrix,ckt->CKTpivotAbsTol);
        if(error != 0) {
            if(error == E_SINGULAR) {
                /* the problem is that the matrix can't be solved with the
                 * current LU factorization.  Maybe if we reload and
                 * try to reorder again it will help...
                 */
                ckt->CKTniState |= NIACSHOULDREORDER;
                goto retry;
            }
            return(error); /* can't handle E_BADMATRIX, so let caller */
        }
    } 
    SMPcSolve(ckt->CKTmatrix,ckt->CKTrhs, 
            ckt->CKTirhs, ckt->CKTrhsSpare,
            ckt->CKTirhsSpare);

    *ckt->CKTrhs = 0;
    *ckt->CKTrhsSpare = 0;
    *ckt->CKTrhsOld = 0;
    *ckt->CKTirhs = 0;
    *ckt->CKTirhsSpare = 0;
    *ckt->CKTirhsOld = 0;

    temp = ckt->CKTirhsOld;
    ckt->CKTirhsOld = ckt->CKTirhs;
    ckt->CKTirhs = temp;

    temp = ckt->CKTrhsOld;
    ckt->CKTrhsOld = ckt->CKTrhs;
    ckt->CKTrhs = temp;
    return(OK);
}
