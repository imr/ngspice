/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

/* 
 * This routine performs the DC and Transient sensitivity
 * calculations
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
CKTsenDCtran(ckt)
register CKTcircuit *ckt;
{
    int error;

#ifdef SENSDEBUG
    printf("time = %.7e\n",ckt->CKTtime);
    printf("CKTsenDCtran\n");
#endif /* SENSDEBUG */

    if(ckt->CKTsenInfo && (ckt->CKTsenInfo->SENmode & TRANSEN) && 
            (ckt->CKTmode & MODEINITTRAN)){

        error = CKTsenLoad(ckt);
        if(error) return(error);
#ifdef SENSDEBUG
        printf("after inittran senload\n");
#endif /* SENSDEBUG */
        error = CKTsenUpdate(ckt);
        if(error) return(error);
#ifdef SENSDEBUG
        printf("after inittran senupdate\n");
#endif /* SENSDEBUG */
    }
    if(ckt->CKTsenInfo && (ckt->CKTsenInfo->SENmode & TRANSEN)&& 
            !(ckt->CKTmode&MODETRANOP))
        ckt->CKTmode = (ckt->CKTmode&(~INITF))|MODEINITFLOAT;

    error = CKTsenLoad(ckt);
    if(error) return(error);

#ifdef SENSDEBUG
    printf("after CKTsenLoad\n");
#endif /* SENSDEBUG */

    error = CKTsenComp(ckt);
    if(error) return(error);

#ifdef SENSDEBUG
    printf("after CKTsenComp\n");
#endif /* SENSDEBUG */

    if(ckt->CKTsenInfo && (ckt->CKTsenInfo->SENmode&TRANSEN) ){
        error = CKTsenUpdate(ckt);
        if(error) return(error);

#ifdef SENSDEBUG
        printf("after CKTsenUpdate\n");
#endif /* SENSDEBUG */

    }


    return(OK);
}
