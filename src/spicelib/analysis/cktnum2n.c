/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

    /* CKTnum2nod
     *  find the given node given its name and return the node pointer
     */

#include "ngspice/ngspice.h"
#include "ngspice/ifsim.h"
#include "ngspice/sperror.h"
#include "ngspice/cktdefs.h"



/* ARGSUSED */
CKTnode *
CKTnum2nod(CKTcircuit *ckt, int node)
{
    CKTnode *here;

    for (here = ckt->CKTnodes; here; here = here->next)  {
        if(here->number == node) {
            return(here);
        }
    }
    return(NULL);
}
