/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

    /* CKTfndTask
     *  find the specified task - not yet supported in spice 
     */

#include "ngspice.h"
#include "cktdefs.h"
#include "ifsim.h"
#include "sperror.h"



/* ARGSUSED */
int
CKTfndTask(void *ckt, void **taskPtr, IFuid taskName)
{
    return(E_UNSUPP);
}
