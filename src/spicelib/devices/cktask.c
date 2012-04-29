/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

/* CKTask
 *
 * Ask questions about a specified device.  */

#include "ngspice/ngspice.h"
#include "ngspice/config.h"
#include "ngspice/devdefs.h"
#include "ngspice/sperror.h"

#include "dev.h"
#include "error.h"

extern bool ft_stricterror;

int
CKTask(CKTcircuit *ckt, GENinstance *instance, int which, IFvalue *value, IFvalue *selector)
{
    int type = instance->GENmodPtr->GENmodType;
    int error;
#ifdef PARALLEL_ARCH
    long msgtype, length;
    long from = instance->GENowner;
#endif /* PARALLEL_ARCH */
    SPICEdev **DEVices;

    DEVices = devices();
    if(DEVices[type]->DEVask) {
        error = DEVices[type]->DEVask(ckt,
                instance, which, value, selector);
    } else {
	error = E_BADPARM;
    }
#ifdef PARALLEL_ARCH
    msgtype = MT_ASK;
    length = sizeof(IFvalue);
    BRDCST_(&msgtype, (char *)value,  &length, &from);
    msgtype++;
    length = sizeof(int);
    BRDCST_(&msgtype, (char *)&error, &length, &from);
#endif /* PARALLEL_ARCH */
    if (ft_stricterror) {
        fprintf(stderr, "\nError: %s\n", errMsg);
        controlled_exit(EXIT_BAD);
    }
    return(error);
}
