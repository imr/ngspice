/**********
Copyright 1993 Regents of the University of California.  All rights reserved.
Author: 1993 David A. Gates
**********/
/*
 */
#ifndef ARCH
#define ARCH

#ifdef PARALLEL_ARCH
#include "sndrcv.h"
#include "evlog.h"

#define MT_LOAD		100
#define MT_ACLOAD	200
#define MT_PZLOAD	300
#define MT_TRANAN	400
#define MT_TRUNC	500
#define MT_COMBINE	600
#define MT_CONV		700
#define MT_ASK		800
#endif /* PARALLEL_ARCH */

extern int ARCHme;	/* My logical process number */
extern int ARCHsize;	/* Total number of processes */

#endif /* ARCH */
