/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

    /* NIdestroy(ckt)
     * delete the data structures allocated for numeric integration.
     */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"


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
        if(ckt->CKTrhsOp) FREE(ckt->CKTrhsOp);
        if(ckt->CKTsenRhs) FREE(ckt->CKTsenRhs);
        if(ckt->CKTseniRhs) FREE(ckt->CKTseniRhs);
        SENdestroy(ckt->CKTsenInfo);
    }
#endif
#ifdef PREDICTOR
    if(ckt->CKTpred) FREE(ckt->CKTpred);
    for(int i=0;i<8;i++) {
        if(ckt->CKTsols[i]) FREE(ckt->CKTsols[i]);
    }
#endif
}
