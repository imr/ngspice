/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

    /* NIdestroy(ckt)
     * delete the data structures allocated for numeric integration.
     */

#include "ngspice.h"
#include "cktdefs.h"
#include "nidest.h"


void
NIdestroy(CKTcircuit *ckt)
{
    if (ckt->CKTmatrix)
	SMPdestroy(ckt->CKTmatrix);
    ckt->CKTmatrix = NULL;
    if(ckt->CKTrhs)         FREE(ckt->CKTrhs);
    if(ckt->CKTrhsOld)      FREE(ckt->CKTrhsOld);
    if(ckt->CKTrhsSpare)    FREE(ckt->CKTrhsSpare);
    if(ckt->CKTirhs)        FREE(ckt->CKTirhs);
    if(ckt->CKTirhsOld)     FREE(ckt->CKTirhsOld);
    if(ckt->CKTirhsSpare)   FREE(ckt->CKTirhsSpare);
#ifdef WANT_SENSE2
    if(ckt->CKTsenInfo){
        if(ckt->CKTrhsOp) FREE(((CKTcircuit *)ckt)->CKTrhsOp);
        if(ckt->CKTsenRhs) FREE(((CKTcircuit *)ckt)->CKTsenRhs);
        if(ckt->CKTseniRhs) FREE(((CKTcircuit *)ckt)->CKTseniRhs);
        SENdestroy(ckt->CKTsenInfo);
    }
#endif
}
