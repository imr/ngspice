/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#ifndef DCOP
#define DCOP


#include "cktdefs.h"
#include "jobdefs.h"
#include "tskdefs.h"
    /*
     * structures used to describe D.C. operationg point analyses to
     * be performed.
     */

typedef struct {
    int JOBtype;
    JOB *JOBnextJob;
    char *JOBname;
} OP;

extern int DCOsetParm(CKTcircuit *ckt, JOB *anal, int which, IFvalue *value);
extern int DCOaskQuest(CKTcircuit *ckt, JOB *anal, int which, IFvalue *value);
#endif /*DCOP*/
