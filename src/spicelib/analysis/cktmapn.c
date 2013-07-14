/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

    /* CKTmapNode(ckt,node)
     *  map the given node to the compact node numbering set of the
     * specified circuit
     */

#include "ngspice/ngspice.h"
#include "ngspice/ifsim.h"
#include "ngspice/sperror.h"
#include "ngspice/cktdefs.h"



/* ARGSUSED *//* fixme abandoned */
int
CKTmapNode(CKTcircuit *ckt, CKTnode **node, IFuid name)
{
    CKTnode *here;
    int error;
    IFuid uid;
    CKTnode *mynode;

    for (here = ckt->CKTnodes; here; here = here->next)  {
        if(here->name == name) {
            if(node) *node = here;
            return(E_EXISTS);
        }
    }
    /* not found, so must be a new one */
    error = CKTmkNode(ckt,&mynode); /*allocate the node*/
    if(error) return(error);
    /* get a uid for it */
    error = SPfrontEnd->IFnewUid (ckt, &uid, NULL, name, UID_SIGNAL, &mynode);
    if(error) return(error);
    mynode->name = uid;     /* set the info we have */
    mynode->type = SP_VOLTAGE;
    error = CKTlinkEq(ckt,mynode); /* and link it in */
    if(node) *node = mynode; /* and finally, return it */
    return(OK);
}
