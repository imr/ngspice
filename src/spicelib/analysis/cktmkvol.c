/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

/* CKTmkVolt
 *  make the given name a 'node' of type voltage in the 
 * specified circuit
 */

#include "ngspice/ngspice.h"
#include "ngspice/ifsim.h"
#include "ngspice/sperror.h"
#include "ngspice/cktdefs.h"



/* ARGSUSED */
int
CKTmkVolt(CKTcircuit *ckt, CKTnode **node, IFuid basename, char *suffix)
{
    IFuid uid;
    int error;
    CKTnode *mynode;
    CKTnode *checknode;

    error = CKTmkNode(ckt,&mynode);
    if(error) return(error);
    checknode = mynode;
    error = SPfrontEnd->IFnewUid (ckt, &uid, basename, suffix, UID_SIGNAL, &checknode);
    if(error) {
        FREE(mynode);
        if(node) *node = checknode;
        return(error);
    }
    mynode->name = uid;
    mynode->type = SP_VOLTAGE;
    if(node) *node = mynode;
    error = CKTlinkEq(ckt,mynode);
    return(error);
}
