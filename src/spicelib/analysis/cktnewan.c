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
CKTnewAnal(void *ckt, int type, IFuid name, void **analPtr, void *taskPtr)
{
    if(type==0) {
        /* special case for analysis type 0 == option card */
        *analPtr=taskPtr; /* pointer to the task itself */
        (*(JOB **)analPtr)->JOBname = name;
        (*(JOB **)analPtr)->JOBtype = type;
        return(OK); /* doesn't need to be created */
    }
    *analPtr = (void *)MALLOC(analInfo[type]->size);
    if(*analPtr==NULL) return(E_NOMEM);
    (*(JOB **)analPtr)->JOBname = name;
    (*(JOB **)analPtr)->JOBtype = type;
    (*(JOB **)analPtr)->JOBnextJob = ((TSKtask *)taskPtr)->jobs;
    ((TSKtask *)taskPtr)->jobs = (JOB *)*analPtr;
    return(OK);
}
