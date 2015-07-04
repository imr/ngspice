/**********
Author: Francesco Lannutti - July 2015
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
    double RELANfinalTime ;
    double RELANstep ;
    double RELANmaxStep ;
    double RELANinitTime ;
    long RELANmode ;
    runDesc *RELANplot ;
} RELANan ;

#define RELAN_TSTART 1
#define RELAN_TSTOP 2
#define RELAN_TSTEP 3
#define RELAN_TMAX 4
#define RELAN_UIC 5
#endif
