/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

    /* CKTfndNode
     *  find the given node given its name and return the node pointer
     */

#include "ngspice.h"
#include "ifsim.h"
#include "sperror.h"
#include "cktdefs.h"



/* ARGSUSED */
int
CKTfndNode(CKTcircuit *ckt, CKTnode **node, IFuid name)
{
    CKTnode *here;

    for (here = ckt->CKTnodes; here; here = here->next)  {
        if(here->name == name) {
            if(node) *node = here;
            return(OK);
        }
    }
    return(E_NOTFOUND);
}
