/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#ifndef ngspice_ACDEFS_H
#define ngspice_ACDEFS_H

#include "ngspice/jobdefs.h"

    /* structure used to describe an AC analysis to be performed */

typedef struct {
    int JOBtype;
    JOB *JOBnextJob;    /* pointer to next thing to do */
    char *JOBname;      /* name of this job */
    double ACstartFreq;
    double ACstopFreq;
    double ACfreqDelta; /* multiplier for decade/octave stepping, */
                        /* step for linear steps. */
    double ACsaveFreq;  /* frequency at which we left off last time*/
    int ACstepType;     /* values described below */
    int ACnumberSteps;
} ACAN;

/* available step types: XXX should be somewhere else */

#define DECADE 1
#define OCTAVE 2
#define LINEAR 3

#define AC_DEC 1
#define AC_OCT 2
#define AC_LIN 3
#define AC_START 4
#define AC_STOP 5
#define AC_STEPS 6

#endif
