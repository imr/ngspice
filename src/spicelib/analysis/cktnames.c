/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1988 Thomas L. Quarles
**********/

    /*
     * CKTnames(ckt)
     *  output information on all circuit nodes/equations
     *
     */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/ifsim.h"
#include "ngspice/iferrmsg.h"


int
CKTnames(CKTcircuit *ckt, int *numNames, IFuid **nameList)
{
    CKTnode *here;
    int i;
    *numNames = ckt->CKTmaxEqNum-1;
    *nameList = TMALLOC(IFuid, *numNames);
    if (*numNames && (*nameList) == NULL) return(E_NOMEM);
    i=0;
    for (here = ckt->CKTnodes->next; here; here = here->next)  {
        (*nameList) [i++] = here->name;
    }
    return(OK);
}

int
CKTdnames(CKTcircuit *ckt)
{
    CKTnode *here;

    for (here = ckt->CKTnodes->next; here; here = here->next)  {
        printf("%03d: %s\n", here->number, here->name);
    }
    return(OK);
}
