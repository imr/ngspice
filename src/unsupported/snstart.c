/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#include "spice.h"
#include <stdio.h>
#include "ifsim.h"
#include "cktdefs.h"
#include "util.h"
#include "const.h"
#include "sperror.h"
#include "suffix.h"

/* This is a routine to initialize the sensitivity
 * data structure
 */

int
SENstartup(ckt)
    CKTcircuit *ckt;
{
    int i;
    int err;
    IFvalue parmtemp;
    int type;
    GENinstance *fast;

#ifdef SENSDEBUG 
    printf("SENstartup\n");
#endif /* SENSDEBUG */ 
    ckt->CKTsenInfo->SENstatus = NORMAL;
    ckt->CKTsenInfo->SENpertfac = 1e-4;
    ckt->CKTsenInfo->SENinitflag = ON;/* allocate memory in
    NIsenReinit */

    parmtemp.iValue = 1;
    for(i=0;i<ckt->CKTsenInfo->SENnumVal;i++) {
        type = -1;
        fast = (GENinstance *)NULL;
        err = CKTfndDev((GENERIC*)ckt,&type,(GENERIC**)&fast,
            (*((ckt->CKTsenInfo->SENdevices)+i)), 
            (GENERIC *)NULL, (GENERIC *)NULL);
        if(err != OK) return(err);
        err = CKTpName(
        (*((ckt->CKTsenInfo->SENparmNames)+i)),
            &parmtemp,ckt ,type,
            (*((ckt->CKTsenInfo->SENdevices)+i)),
            &fast);
        if(err != OK) return(err);
    }
#ifdef SENSDEBUG 
    printf("SENstartup end\n");
#endif /* SENSDEBUG */ 
    return(OK);
}
