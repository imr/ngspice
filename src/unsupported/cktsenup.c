/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

/* CKTsenUpdate(ckt)
 * this is a driver program to iterate through all the various
 * sensitivity update functions provided for the circuit elements 
 * in the given circuit 
 */

#include "spice.h"
#include <stdio.h>
#include "smpdefs.h"
#include "cktdefs.h"
#include "devdefs.h"
#include "sperror.h"
#include "trandefs.h"
#include "suffix.h"


int
CKTsenUpdate(ckt)
register CKTcircuit *ckt;
{
    extern SPICEdev *DEVices[];
    register int i;
    int error;


    for (i=0;i<DEVmaxnum;i++) {
        if ( ((*DEVices[i]).DEVsenUpdate != NULL) 
            && (ckt->CKThead[i] != NULL) ){
            error = (*((*DEVices[i]).DEVsenUpdate))(ckt->CKThead[i],ckt);
            if(error) return(error);
        }
    }
    return(OK);
}
