/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

    /* CKTask
     *  Ask questions about a specified device.
     */

#include "ngspice.h"
#include <stdio.h>
#include "cktdefs.h"
#include "ifsim.h"
#include "devdefs.h"
#include "sperror.h"


extern SPICEdev *DEVices[];

int
CKTask(void *ckt, void *fast, int which, IFvalue *value, IFvalue *selector)
{
    register int type = ((GENinstance *)fast)->GENmodPtr->GENmodType;
    int error;
#ifdef PARALLEL_ARCH
    long msgtype, length;
    long from = ((GENinstance *)fast)->GENowner;
#endif /* PARALLEL_ARCH */

    if((*DEVices[type]).DEVask) {
        error = DEVices[type]->DEVask((CKTcircuit *)ckt,
                (GENinstance *)fast,which,value,selector);
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
    return(error);
}
