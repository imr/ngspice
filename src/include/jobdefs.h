/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#ifndef JOBdef
#define JOBdef


#include "ifsim.h"

typedef struct sJOB{
    int JOBtype;                /* type of job */
    struct sJOB *JOBnextJob;    /* next job in list */
    IFuid JOBname;              /* name of this job */

} JOB;

typedef struct {
    IFanalysis public;
    int size;
    int domain;
    int do_ic;
    int (*(setParm))( );
    int (*(askQuest))( );
    int (*an_init)( );
    int (*an_func)( );
} SPICEanalysis;

#define NODOMAIN	0
#define TIMEDOMAIN	1
#define FREQUENCYDOMAIN 2
#define SWEEPDOMAIN	3

#endif /*JOBdef*/
