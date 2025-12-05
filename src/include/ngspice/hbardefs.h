/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#ifndef ngspice_SPDEFS_H
#define ngspice_SPDEFS_H

#include "ngspice/jobdefs.h"

#ifdef WITH_HB
    /* structure used to describe an AC analysis to be performed */

typedef struct {
    int JOBtype;
    JOB *JOBnextJob;    /* pointer to next thing to do */
    char *JOBname;      /* name of this job */
    double HBFreq1;
    double HBFreq2;
    int SPnoiseInput;
    int SPnoiseOutput;
} HBAN;

/* available step types: XXX should be somewhere else */
#ifndef ngspice_ACDEFS_H
enum {
    DECADE = 1,
    OCTAVE,
    LINEAR,
};
#endif

enum {
    HB_F1 = 1,
    HB_F2
};
#endif
#endif
