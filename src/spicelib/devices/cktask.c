/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

/* CKTask
 *
 * Ask questions about a specified device.  */

#include "ngspice/ngspice.h"
#include "ngspice/devdefs.h"
#include "ngspice/sperror.h"

#include "dev.h"
#include "ngspice/fteext.h" /* controlled_exit() */

int
CKTask(CKTcircuit *ckt, GENinstance *instance, int which, IFvalue *value, IFvalue *selector)
{
    int type = instance->GENmodPtr->GENmodType;
    int error;
    SPICEdev **DEVices;

    DEVices = devices();
    if(DEVices[type]->DEVask) {
        error = DEVices[type]->DEVask(ckt,
                                      instance, which, value, selector);
    } else {
        error = E_BADPARM;
    }
    if (error && ft_stricterror) {
        fprintf(stderr, "\nError: %s\n", errMsg);
        controlled_exit(EXIT_BAD);
    }
    return(error);
}
