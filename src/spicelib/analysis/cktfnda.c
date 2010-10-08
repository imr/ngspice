/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

    /* CKTfndAnal
     *  find the given Analysis given its name and return the Analysis pointer
     */

#include "ngspice.h"
#include "ifsim.h"
#include "jobdefs.h"
#include "tskdefs.h"
#include "sperror.h"
#include "cktdefs.h"



/* ARGSUSED */
int
CKTfndAnal(CKTcircuit *ckt, int *analIndex, JOB **anal, IFuid name, TSKtask *inTask, IFuid taskName)
{
    TSKtask *task = /* fixme, drop that */ inTask;
    JOB *here;

    for (here = task->jobs; here; here = here->JOBnextJob) {
        if(strcmp(here->JOBname,name)==0) {
            if(anal) *anal = here;
            return(OK);
        }
    }
    return(E_NOTFOUND);
}
