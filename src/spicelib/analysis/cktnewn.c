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

#include "ngspice.h"
#include "ifsim.h"
#include "iferrmsg.h"
#include "smpdefs.h"
#include "cktdefs.h"


/* should just call CKTnewEQ and set node type afterwards */
int
CKTnewNode(void *inCkt, void **node, IFuid name)
{
    CKTcircuit *ckt = (CKTcircuit *)inCkt;
    if(!(ckt->CKTnodes)) { /*  starting the list - allocate both ground and 1 */
        ckt->CKTnodes = (CKTnode *) MALLOC(sizeof(CKTnode));
        if(ckt->CKTnodes == (CKTnode *)NULL) return(E_NOMEM);
        ckt->CKTnodes->name = (char *)NULL;
        ckt->CKTnodes->type = SP_VOLTAGE;
        ckt->CKTnodes->number = 0;
        ckt->CKTlastNode = ckt->CKTnodes;
    }
    ckt->CKTlastNode->next = (CKTnode *)MALLOC(sizeof(CKTnode));
    if(ckt->CKTlastNode->next == (CKTnode *)NULL) return(E_NOMEM);
    ckt->CKTlastNode = ckt->CKTlastNode->next;
    ckt->CKTlastNode->name = name;
    ckt->CKTlastNode->number = ckt->CKTmaxEqNum++;
    ckt->CKTlastNode->type = SP_VOLTAGE;
    ckt->CKTlastNode->next = (CKTnode *)NULL;

    if(node) *node = (void *)ckt->CKTlastNode;
    return(OK);
}
