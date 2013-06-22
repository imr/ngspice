/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

    /*
     *CKTmkNode(ckt,node)
     *  Tentatively allocate a new circuit equation structure
     */

#include "ngspice/ngspice.h"
#include "ngspice/ifsim.h"
#include "ngspice/iferrmsg.h"
#include "ngspice/smpdefs.h"
#include "ngspice/cktdefs.h"


/* ARGSUSED */
int
CKTmkNode(CKTcircuit *ckt, CKTnode **node)
{
    CKTnode *mynode;

    NG_IGNORE(ckt);

    mynode = TMALLOC(CKTnode, 1);
    if(mynode == NULL) return(E_NOMEM);
    mynode->next = NULL;
    mynode->name = NULL;

    if(node) *node = mynode;
    return(OK);
}
