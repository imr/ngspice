/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

/* CKTsenUpdate(ckt)
 * this is a driver program to iterate through all the various
 * sensitivity update functions provided for the circuit elements
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
CKTsenUpdate(CKTcircuit *ckt)
{
    int i;
    int error;

    for (i = 0; i < DEVmaxnum; i++)
        if (DEVices[i] && DEVices[i]->DEVsenUpdate && ckt->CKThead[i]) {
            error = DEVices[i]->DEVsenUpdate (ckt->CKThead[i], ckt);
            if (error)
                return error;
        }

    return OK;
}
