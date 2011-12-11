/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

    /* CKTfndTask
     *  find the specified task - not yet supported in spice 
     */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/ifsim.h"
#include "ngspice/sperror.h"



/* ARGSUSED */
int
CKTfndTask(CKTcircuit *ckt, TSKtask **taskPtr, IFuid taskName)
{
    NG_IGNORE(ckt);
    NG_IGNORE(taskPtr);
    NG_IGNORE(taskName);

    return(E_UNSUPP);
}
