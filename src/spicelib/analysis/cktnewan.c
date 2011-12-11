/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include "ngspice/tskdefs.h"
#include "ngspice/jobdefs.h"
#include "ngspice/ifsim.h"
#include "ngspice/iferrmsg.h"
#include "ngspice/cktdefs.h"

#include "analysis.h"

extern SPICEanalysis *analInfo[];

/* ARGSUSED */
int
CKTnewAnal(CKTcircuit *ckt, int type, IFuid name, JOB **analPtr, TSKtask *taskPtr)
{
    NG_IGNORE(ckt);

    if(type==0) {
        /* special case for analysis type 0 == option card */
        *analPtr = &(taskPtr->taskOptions); /* pointer to the task itself */
        (*analPtr)->JOBname = name;
        (*analPtr)->JOBtype = type;
        return(OK); /* doesn't need to be created */
    }
    *analPtr = (JOB *) tmalloc((size_t) analInfo[type]->size);
    if(*analPtr==NULL) return(E_NOMEM);
    (*analPtr)->JOBname = name;
    (*analPtr)->JOBtype = type;
    (*analPtr)->JOBnextJob = taskPtr->jobs;
    taskPtr->jobs = *analPtr;
    return(OK);
}
