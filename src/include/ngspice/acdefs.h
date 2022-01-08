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

#ifdef RFSPICE
#ifndef ngspice_SPDEFS_H
enum {
    DECADE = 1,
    OCTAVE,
    LINEAR,
};
#endif
#else
enum {
    DECADE = 1,
    OCTAVE,
    LINEAR,
};
#endif

enum {
    AC_DEC = 1,
    AC_OCT,
    AC_LIN,
    AC_START,
    AC_STOP,
    AC_STEPS,
};

#endif
