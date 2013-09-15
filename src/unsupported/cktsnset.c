/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/


/*
 * CKTsenSetup(ckt)
 * this is a driver program to iterate through all the various
 * sensitivity setup functions provided for the circuit elements
 * in the given circuit
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
CKTsenSetup(CKTcircuit *ckt)
{
    int i;
    int error;
    SENstruct *info;

#ifdef SENSDEBUG
    printf("CKTsenSetup\n");
#endif

    info = ckt->CKTsenInfo;
    info->SENparms = 0;

    for (i = 0; i < DEVmaxnum; i++)
        if (DEVices[i] && DEVices[i]->DEVsenSetup && ckt->CKThead[i]) {
            error = DEVices[i]->DEVsenSetup (info, ckt->CKThead[i]);
            if (error)
                return error;
        }

#ifdef SENSDEBUG
    printf("CKTsenSetup end\n");
#endif

    return OK;
}
