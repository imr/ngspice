/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

    /* CKTground(ckt,node)
     *  specify the node to be the ground node of the given circuit
     */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/ifsim.h"
#include "ngspice/sperror.h"



int
CKTground(CKTcircuit *ckt, CKTnode **node, IFuid name)
{
    if(ckt->CKTnodes) {
        if(ckt->CKTnodes->name) {
            /*already exists - keep old name, but return it */
            if(node) *node = ckt->CKTnodes;
            return(E_EXISTS);
        }
        ckt->CKTnodes->name = name;
        ckt->CKTnodes->type = SP_VOLTAGE;
        ckt->CKTnodes->number = 0;
    } else {
        ckt->CKTnodes = TMALLOC(CKTnode, 1);
        if(ckt->CKTnodes == NULL) return(E_NOMEM);
        ckt->CKTnodes->name = name;
        ckt->CKTnodes->type = SP_VOLTAGE;
        ckt->CKTnodes->number = 0;
        ckt->CKTnodes->next = NULL;
        ckt->CKTlastNode = ckt->CKTnodes;
    }
    if(node) *node = ckt->CKTnodes;
    return(OK);

}
