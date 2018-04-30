/**********
Author: 2010-05 Stefano Perticaroli ``spertica''
Review: 2012-10 Francesco Lannutti
**********/

#ifndef ngspice_PSSDEFS_H
#define ngspice_PSSDEFS_H

#include "ngspice/jobdefs.h"
#include "ngspice/tskdefs.h"
    /*
     * PSSdefs.h - defs for pss analyses
     */

typedef struct {
    int JOBtype;
    JOB *JOBnextJob;
    char *JOBname;
    double PSSguessedFreq;
    CKTnode *PSSoscNode;
    double PSSstabTime;
    long PSSmode;
    long int PSSpoints;
    int PSSharms;
    runDesc *PSSplot_td;
    runDesc *PSSplot_fd;
    int sc_iter;
    double steady_coeff;
} PSSan;

enum {
    GUESSED_FREQ = 1,
    STAB_TIME,
    OSC_NODE,
    PSS_POINTS,
    PSS_HARMS,
    PSS_UIC,
    SC_ITER,
    STEADY_COEFF,
};

#endif
