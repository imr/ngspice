/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

    /* CKTdump(ckt)
     * this is a simple program to dump the rhs vector to stdout
     */

#include "ngspice.h"
#include <stdio.h>
#include "smpdefs.h"
#include "cktdefs.h"



void
CKTdump(CKTcircuit *ckt, double ref, void *plot)
{
    IFvalue refData;
    IFvalue valData;

    refData.rValue = ref;
    valData.v.numValue = ckt->CKTmaxEqNum-1;
    valData.v.vec.rVec = ckt->CKTrhsOld+1;
    (*(SPfrontEnd->OUTpData))(plot,&refData,&valData);
}
