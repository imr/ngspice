/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

    /*
     *CKTlinkEq
     *  Link an already allocated node into the necessary structure
     */

#include "ngspice/ngspice.h"
#include "ngspice/ifsim.h"
#include "ngspice/iferrmsg.h"
#include "ngspice/smpdefs.h"
#include "ngspice/cktdefs.h"


int
CKTlinkEq(CKTcircuit *ckt, CKTnode *node)
{
    if(!(ckt->CKTnodes)) { /*  starting the list - allocate both ground and 1 */
        ckt->CKTnodes = TMALLOC(CKTnode, 1);
        if(ckt->CKTnodes == NULL) return(E_NOMEM);
        ckt->CKTnodes->name = NULL;
        ckt->CKTnodes->type = SP_VOLTAGE;
        ckt->CKTnodes->number = 0;
        ckt->CKTlastNode = ckt->CKTnodes;
    }
    if(node == NULL) return(E_BADPARM);
    ckt->CKTlastNode->next = node;
    ckt->CKTlastNode = ckt->CKTlastNode->next;
    ckt->CKTlastNode->number = ckt->CKTmaxEqNum++;
    ckt->CKTlastNode->next = NULL;
    return(OK);
}
