/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

#include "spice.h"
#include <stdio.h>
#include "smpdefs.h"
#include "cktdefs.h"
#include "devdefs.h"
#include "sperror.h"
#include "trandefs.h"
#include "suffix.h"


    /* CKTsenPrint(ckt)
     * this is a driver program to iterate through all the
     * various sensitivity print functions provided for 
     * the circuit elements in the given circuit 
     */

void
CKTsenPrint(ckt)
register CKTcircuit *ckt;
{
    extern SPICEdev *DEVices[];
    register int i;

    for (i=0;i<DEVmaxnum;i++) {
        if ( ((*DEVices[i]).DEVsenPrint != NULL) && (ckt->CKThead[i] != NULL) ){
            (*((*DEVices[i]).DEVsenPrint))(ckt->CKThead[i],ckt);
        }
    }
}
