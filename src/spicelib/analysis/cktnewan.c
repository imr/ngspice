/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#include "ngspice.h"
#include "tskdefs.h"
#include "jobdefs.h"
#include "ifsim.h"
#include "iferrmsg.h"
#include <cktdefs.h>

#include "analysis.h"

extern SPICEanalysis *analInfo[];

/* ARGSUSED */
int
CKTnewAnal(CKTcircuit *ckt, int type, IFuid name, JOB **analPtr, TSKtask *taskPtr)
{
    if(type==0) {
        /* special case for analysis type 0 == option card */
        *analPtr = &(taskPtr->taskOptions); /* pointer to the task itself */
        (*analPtr)->JOBname = name;
        (*analPtr)->JOBtype = type;
        return(OK); /* doesn't need to be created */
    }
    *analPtr = (JOB *)MALLOC(analInfo[type]->size);
    if(*analPtr==NULL) return(E_NOMEM);
    (*analPtr)->JOBname = name;
    (*analPtr)->JOBtype = type;
    (*analPtr)->JOBnextJob = taskPtr->jobs;
    taskPtr->jobs = *analPtr;
    return(OK);
}
