/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

    /*
     *CKTnodName(ckt)
     *  output information on all circuit nodes/equations
     *
     */

#include "ngspice.h"
#include <stdio.h>
#include "cktdefs.h"



IFuid
CKTnodName(CKTcircuit *ckt, register int nodenum)
{
    register CKTnode *here;

    for(here = ckt->CKTnodes;here; here = here->next) {
        if(here->number == nodenum) { 
            /* found it */
            return(here->name);
        }
    }
    /* doesn't exist - do something */
    return("UNKNOWN NODE");
}
