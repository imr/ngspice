/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

    /* CKTdltInst
     *  delete the specified instance - not yet supported in spice 
     */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/ifsim.h"
#include "ngspice/sperror.h"



/* ARGSUSED */
int
CKTdltInst(CKTcircuit *ckt, void *instance)
{
    NG_IGNORE(ckt);
    NG_IGNORE(instance);

    return(E_UNSUPP);
}
