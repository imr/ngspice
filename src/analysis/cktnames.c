/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1988 Thomas L. Quarles
**********/

    /*
     * CKTnames(ckt)
     *  output information on all circuit nodes/equations
     *
     */

#include "ngspice.h"
#include <stdio.h>
#include "cktdefs.h"
#include "ifsim.h"
#include "iferrmsg.h"


int
CKTnames(register CKTcircuit *ckt, int *numNames, register IFuid **nameList)
{
    register CKTnode *here;
    register int i;
    *numNames = ckt->CKTmaxEqNum-1;
    *nameList = (IFuid *)MALLOC(*numNames * sizeof(IFuid ));
    if ((*nameList) == (IFuid *)NULL) return(E_NOMEM);
    i=0;
    for (here = ckt->CKTnodes->next; here; here = here->next)  {
        *((*nameList)+i++) = here->name;
    }
    return(OK);
}

int
CKTdnames(CKTcircuit *ckt)
{
    CKTnode *here;
    register int i;

    i=0;
    for (here = ckt->CKTnodes->next; here; here = here->next)  {
        printf("%03d: %s\n", here->number, (char *)here->name);
    }
    return(OK);
}
