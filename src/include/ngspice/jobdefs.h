/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#ifndef ngspice_JOBDEFS_H
#define ngspice_JOBDEFS_H


#include "ngspice/typedefs.h"
#include "ngspice/ifsim.h"

struct JOB {
    int JOBtype;                /* type of job */
    struct JOB *JOBnextJob;     /* next job in list */
    IFuid JOBname;              /* name of this job */
};

#define NODOMAIN	0
#define TIMEDOMAIN	1
#define FREQUENCYDOMAIN 2
#define SWEEPDOMAIN	3

#endif
