/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

    /* CKTdestroy(ckt)
     * this is a driver program to iterate through all the various
     * destroy functions provided for the circuit elements in the
     * given circuit 
     */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/devdefs.h"
#include "ngspice/ifsim.h"
#include "ngspice/sperror.h"

#ifdef XSPICE
#include "ngspice/evtproto.h"
#endif

int
CKTdestroy(CKTcircuit *ckt)
{
    int i;
    CKTnode *node;
    CKTnode *nnode;


#ifdef WANT_SENSE2
    if(ckt->CKTsenInfo){
         if(ckt->CKTrhsOp) FREE(ckt->CKTrhsOp);
         if(ckt->CKTsenRhs) FREE(ckt->CKTsenRhs);
         if(ckt->CKTseniRhs) FREE(ckt->CKTseniRhs);
         SENdestroy(ckt->CKTsenInfo);
    }
#endif

    for (i=0;i<DEVmaxnum;i++) {
        if ( DEVices[i] && DEVices[i]->DEVdestroy && ckt->CKThead[i] ) {
            DEVices[i]->DEVdestroy (&(ckt->CKThead[i]));
        }
    }
    for(i=0;i<=ckt->CKTmaxOrder+1;i++){
        FREE(ckt->CKTstates[i]);
    }
    if(ckt->CKTmatrix) {
        SMPdestroy(ckt->CKTmatrix);
        ckt->CKTmatrix = NULL;
    }
    FREE(ckt->CKTbreaks);
    for(node = ckt->CKTnodes; node; ) {
        nnode = node->next;
        FREE(node);
        node = nnode;
    }
    ckt->CKTnodes = NULL;
    ckt->CKTlastNode = NULL;

    FREE(ckt->CKTrhs);
    FREE(ckt->CKTrhsOld);
    FREE(ckt->CKTrhsSpare);
    FREE(ckt->CKTirhs);
    FREE(ckt->CKTirhsOld);
    FREE(ckt->CKTirhsSpare);

    FREE(ckt->CKTstat->STATdevNum);
    FREE(ckt->CKTstat);
    FREE(ckt->CKThead);

#ifdef XSPICE
    EVTdest(ckt->evt);
    FREE(ckt->enh);
    FREE(ckt->evt);
#endif

    nghash_free(ckt->DEVnameHash, NULL, NULL);
    nghash_free(ckt->MODnameHash, NULL, NULL);
    FREE(ckt);
    return(OK);
}
