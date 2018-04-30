/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/


#ifndef ngspice_TRANDEFS_H
#define ngspice_TRANDEFS_H


#include "ngspice/cktdefs.h"
#include "ngspice/jobdefs.h"
#include "ngspice/tskdefs.h"
    /*
     * TRANdefs.h - defs for transient analyses 
     */

typedef struct {
    int JOBtype;
    JOB *JOBnextJob;
    char *JOBname;
    double TRANfinalTime;
    double TRANstep;
    double TRANmaxStep;
    double TRANinitTime;
    long TRANmode;
    runDesc *TRANplot;
} TRANan;

enum {
    TRAN_TSTART = 1,
    TRAN_TSTOP,
    TRAN_TSTEP,
    TRAN_TMAX,
    TRAN_UIC,
};

#endif
