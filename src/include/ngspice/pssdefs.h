/**********
Author: 2010-05 Stefano Perticaroli ``spertica''
Review: 2012-10 Francesco Lannutti
**********/

#ifndef PSS_H
#define PSS_H

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

#define GUESSED_FREQ 1
#define STAB_TIME 2
#define OSC_NODE 3
#define PSS_POINTS 4
#define PSS_HARMS 5
#define PSS_UIC 6
#define SC_ITER 7
#define STEADY_COEFF 8

#endif
