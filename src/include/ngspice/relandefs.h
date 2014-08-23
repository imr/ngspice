/**********
Author: Francesco Lannutti - August 2014
**********/

#ifndef ngspice_RELANDEFS_H
#define ngspice_RELANDEFS_H

#include "ngspice/cktdefs.h"
#include "ngspice/jobdefs.h"
#include "ngspice/tskdefs.h"

typedef struct {
    int JOBtype ;
    JOB *JOBnextJob ;
    char *JOBname ;
    double RELANagingStep ;
    double RELANagingTotalTime ;
    double RELANagingStartTime ;
    long RELANmode ;
    runDesc *RELANplot ;
} RELANan ;

#define RELAN_AGING_STEP  1
#define RELAN_AGING_STOP  2
#define RELAN_AGING_START 3
#define RELAN_UIC         4

#endif
