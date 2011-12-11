/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

    /*
     *CKTnewNode(ckt,node,name)
     *  Allocate a new circuit equation number (returned) in the specified
     *  circuit to contain a new equation or node
     * returns -1 for failure to allocate a node number 
     *
     */

#include "ngspice/ngspice.h"
#include "ngspice/ifsim.h"
#include "ngspice/iferrmsg.h"
#include "ngspice/smpdefs.h"
#include "ngspice/cktdefs.h"


/* should just call CKTnewEQ and set node type afterwards */
int
CKTnewNode(CKTcircuit *ckt, CKTnode **node, IFuid name)
{
    if(!(ckt->CKTnodes)) { /*  starting the list - allocate both ground and 1 */
        ckt->CKTnodes = TMALLOC(CKTnode, 1);
        if(ckt->CKTnodes == NULL) return(E_NOMEM);
        ckt->CKTnodes->name = NULL;
        ckt->CKTnodes->type = SP_VOLTAGE;
        ckt->CKTnodes->number = 0;
        ckt->CKTlastNode = ckt->CKTnodes;
    }
    ckt->CKTlastNode->next = TMALLOC(CKTnode, 1);
    if(ckt->CKTlastNode->next == NULL) return(E_NOMEM);
    ckt->CKTlastNode = ckt->CKTlastNode->next;
    ckt->CKTlastNode->name = name;
    ckt->CKTlastNode->number = ckt->CKTmaxEqNum++;
    ckt->CKTlastNode->type = SP_VOLTAGE;
    ckt->CKTlastNode->next = NULL;

    if(node) *node = ckt->CKTlastNode;
    return(OK);
}
