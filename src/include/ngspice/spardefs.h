/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#ifndef ngspice_SPDEFS_H
#define ngspice_SPDEFS_H

#include "ngspice/jobdefs.h"

#ifdef RFSPICE
    /* structure used to describe an AC analysis to be performed */

typedef struct {
    int JOBtype;
    JOB *JOBnextJob;    /* pointer to next thing to do */
    char *JOBname;      /* name of this job */
    double SPstartFreq;
    double SPstopFreq;
    double SPfreqDelta; /* multiplier for decade/octave stepping, */
                        /* step for linear steps. */
    double SPsaveFreq;  /* frequency at which we left off last time*/
    int SPstepType;     /* values described below */
    int SPnumberSteps;

    unsigned SPdoNoise : 1; /* Flag to indicate if SP noise must be calculated*/

    int SPnoiseInput;
    int SPnoiseOutput;
} SPAN;

/* available step types: XXX should be somewhere else */
#ifndef ngspice_ACDEFS_H
enum {
    DECADE = 1,
    OCTAVE,
    LINEAR,
};
#endif

enum {
    SP_DEC = 1,
    SP_OCT,
    SP_LIN,
    SP_START,
    SP_STOP,
    SP_STEPS,
    SP_DONOISE,
};
#endif
#endif
