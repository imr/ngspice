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

#include "ngspice/ngspice.h"
#include "ngspice/ifsim.h"
#include "ngspice/iferrmsg.h"
#include "ngspice/smpdefs.h"
#include "ngspice/cktdefs.h"


int
CKTnewEq(CKTcircuit *ckt, CKTnode **node, IFuid name)
{
    CKTnode *mynode;
    int error;

    error = CKTmkNode(ckt,&mynode);
    if(error) return(error);

    if(node) *node = mynode;
    mynode->name = name;

    error = CKTlinkEq(ckt,mynode);

    return(error);
}
