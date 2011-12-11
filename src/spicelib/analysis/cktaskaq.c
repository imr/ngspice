/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include "ngspice/tskdefs.h"
#include "ngspice/jobdefs.h"
#include "ngspice/cktdefs.h"
#include "ngspice/ifsim.h"
#include "ngspice/iferrmsg.h"

#include "analysis.h"

extern SPICEanalysis *analInfo[];

/* ARGSUSED */
int
CKTaskAnalQ(CKTcircuit *ckt, JOB *analPtr, int parm, IFvalue *value, IFvalue *selector)
{
    int type = analPtr->JOBtype;

    NG_IGNORE(selector);

    if((analInfo[type]->askQuest) == NULL) return(E_BADPARM);
    return( analInfo[type]->askQuest (ckt, analPtr, parm, value));
}
