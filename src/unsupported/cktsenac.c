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


/* CKTsenAC(ckt)
 * this is a routine for AC sensitivity calculations
 */

int
CKTsenAC(ckt)
register CKTcircuit *ckt;
{
    int error;

#ifdef SENSDEBUG
    printf("CKTsenAC\n");
#endif /* SENSDEBUG */


    if(error = CKTsenLoad(ckt)) return(error);

#ifdef SENSDEBUG
    printf("after CKTsenLoad\n");
#endif /* SENSDEBUG */

    if(error = CKTsenComp(ckt)) return(error);

#ifdef SENSDEBUG
    printf("after CKTsenComp\n");
#endif /* SENSDEBUG */

    return(OK);
}

