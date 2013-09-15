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


/* CKTsenAC(ckt)
 * this is a routine for AC sensitivity calculations
 */

int
CKTsenAC(CKTcircuit *ckt)
{
    int error;

#ifdef SENSDEBUG
    printf("CKTsenAC\n");
#endif

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

    return OK;
}
