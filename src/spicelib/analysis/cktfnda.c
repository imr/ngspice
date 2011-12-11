/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

    /* CKTfndAnal
     *  find the given Analysis given its name and return the Analysis pointer
     */

#include "ngspice/ngspice.h"
#include "ngspice/ifsim.h"
#include "ngspice/jobdefs.h"
#include "ngspice/tskdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/cktdefs.h"



/* ARGSUSED */
int
CKTfndAnal(CKTcircuit *ckt, int *analIndex, JOB **anal, IFuid name, TSKtask *task, IFuid taskName)
{
    JOB *here;

    NG_IGNORE(ckt);
    NG_IGNORE(analIndex);
    NG_IGNORE(taskName);

    for (here = task->jobs; here; here = here->JOBnextJob) {
        if(strcmp(here->JOBname,name)==0) {
            if(anal) *anal = here;
            return(OK);
        }
    }
    return(E_NOTFOUND);
}
