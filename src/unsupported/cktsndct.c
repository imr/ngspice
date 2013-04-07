/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

/*
 * This routine performs the DC and Transient sensitivity
 * calculations
 */

#include "ngspice/ngspice.h"
#include <stdio.h>
#include "ngspice/smpdefs.h"
#include "ngspice/cktdefs.h"
#include "ngspice/devdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/trandefs.h"
#include "ngspice/suffix.h"


int
CKTsenDCtran(CKTcircuit *ckt)
{
    int error;

#ifdef SENSDEBUG
    printf("time = %.7e\n", ckt->CKTtime);
    printf("CKTsenDCtran\n");
#endif

    if (ckt->CKTsenInfo && (ckt->CKTsenInfo->SENmode & TRANSEN) &&
        (ckt->CKTmode & MODEINITTRAN))
    {
        error = CKTsenLoad(ckt);
        if (error)
            return error;

#ifdef SENSDEBUG
        printf("after inittran senload\n");
#endif
        error = CKTsenUpdate(ckt);
        if (error)
            return error;

#ifdef SENSDEBUG
        printf("after inittran senupdate\n");
#endif

    }

    if (ckt->CKTsenInfo && (ckt->CKTsenInfo->SENmode & TRANSEN) &&
        !(ckt->CKTmode & MODETRANOP))
    {
        ckt->CKTmode = (ckt->CKTmode & (~INITF)) | MODEINITFLOAT;
    }

    error = CKTsenLoad(ckt);
    if (error)
        return error;

#ifdef SENSDEBUG
    printf("after CKTsenLoad\n");
#endif

    error = CKTsenComp(ckt);
    if (error)
        return error;

#ifdef SENSDEBUG
    printf("after CKTsenComp\n");
#endif

    if (ckt->CKTsenInfo && (ckt->CKTsenInfo->SENmode & TRANSEN)) {
        error = CKTsenUpdate(ckt);
        if (error)
            return error;

#ifdef SENSDEBUG
        printf("after CKTsenUpdate\n");
#endif

    }

    return OK;
}
