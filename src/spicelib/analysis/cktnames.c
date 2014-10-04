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
    if ((*nameList) == NULL) return(E_NOMEM);
    i=0;
    printf("Analog Nodes:\n");/* holmes : tracking down node name origins. */
    printf("-------------\n");/* holmes : tracking down node name origins. */
    for (here = ckt->CKTnodes->next; here; here = here->next)  {
        printf("%s\n",here->name); /* holmes : tracking down node name origins. */
        (*nameList) [i++] = here->name;
    }
    printf("-------------\n\n");/* holmes : tracking down node name origins. */
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
