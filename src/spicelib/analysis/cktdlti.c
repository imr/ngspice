/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

    /* CKTdltInst
     *  delete the specified instance - not yet supported in spice 
     */

#include "ngspice.h"
#include "cktdefs.h"
#include "ifsim.h"
#include "sperror.h"



/* ARGSUSED */
int
CKTdltInst(void *ckt, void *instance)
{
    return(E_UNSUPP);
}
