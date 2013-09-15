/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include <stdio.h>
#include "ngspice/smpdefs.h"
#include "ngspice/cktdefs.h"
#include "ngspice/devdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/trandefs.h"
#include "ngspice/suffix.h"


/* CKTsenPrint(ckt)
 * this is a driver program to iterate through all the
 * various sensitivity print functions provided for
 * the circuit elements in the given circuit
 */

void
CKTsenPrint(CKTcircuit *ckt)
{
    int i;

    for (i = 0; i < DEVmaxnum; i++)
        if (DEVices[i] && DEVices[i]->DEVsenPrint && ckt->CKThead[i])
            DEVices[i]->DEVsenPrint (ckt->CKThead[i], ckt);
}
