/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

    /*
     *CKTnewEq(ckt,node,name)
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


int
CKTnewEq(void *inCkt, void **node, IFuid name)
{
    CKTnode *mynode;
    CKTcircuit *ckt = (CKTcircuit *)inCkt;
    int error;

    error = CKTmkNode(ckt,&mynode);
    if(error) return(error);

    if(node) *node = (void *)mynode;
    mynode->name = name;

    error = CKTlinkEq(ckt,mynode);

    return(error);
}
