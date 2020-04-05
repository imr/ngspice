/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/smpdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/devdefs.h"


int
CKTic(CKTcircuit *ckt)
{
    int error;
    int size;
    int i;
    CKTnode *node;

    size = SMPmatSize(ckt->CKTmatrix);
    for (i=0;i<=size;i++) {
        ckt->CKTrhs[i]=0;
    }

    for(node = ckt->CKTnodes;node != NULL; node = node->next) {
        if(node->nsGiven) {
            node->ptr = SMPmakeElt(ckt->CKTmatrix,node->number,node->number);
            if(node->ptr == NULL) return(E_NOMEM);
            ckt->CKThadNodeset = 1;
            ckt->CKTrhsOld[node->number] = ckt->CKTrhs[node->number] = node->nodeset;
        }
        if(node->icGiven) {
            if(! ( node->ptr)) {
                node->ptr = SMPmakeElt(ckt->CKTmatrix,node->number,
                        node->number);
                if(node->ptr == NULL) return(E_NOMEM);
            }
            ckt->CKTrhsOld[node->number] = ckt->CKTrhs[node->number] = node->ic;
        }
    }

    if(ckt->CKTmode & MODEUIC) {
        for (i=0;i<DEVmaxnum;i++) {
            if( DEVices[i] && DEVices[i]->DEVsetic && ckt->CKThead[i] ) {
                error = DEVices[i]->DEVsetic (ckt->CKThead[i], ckt);
                if(error) return(error);
            }
        }
    }

    return(OK);
}
