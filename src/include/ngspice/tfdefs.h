/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#ifndef ngspice_TFDEFS_H
#define ngspice_TFDEFS_H

#include "ngspice/typedefs.h"
#include "ngspice/jobdefs.h"
#include "ngspice/tskdefs.h"
#include "ngspice/cktdefs.h"


    /* TFdefs.h - defs for transfer function analyses */

struct TFan {
    int JOBtype;
    JOB *JOBnextJob;
    IFuid JOBname;
    CKTnode *TFoutPos;
    CKTnode *TFoutNeg;
    IFuid TFoutSrc;
    IFuid TFinSrc;
    char *TFoutName;    /* a printable name for an output v(x,y) */
    unsigned int TFoutIsV :1;
    unsigned int TFoutIsI :1;
    unsigned int TFinIsV :1;
    unsigned int TFinIsI :1;
};


enum {
    TF_OUTPOS = 1,
    TF_OUTNEG,
    TF_OUTSRC,
    TF_INSRC,
    TF_OUTNAME,
};

#endif
