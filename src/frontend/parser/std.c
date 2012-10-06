/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Wayne A. Christopher, U. C. Berkeley CAD Group
**********/

/*
 * Standard utility routines.
 * Most moved to MISC/
 */

#include "ngspice/ngspice.h"
#include "ngspice/cpstd.h"


/* This might not be around.  If not then forget about sorting. */

#ifndef HAVE_QSORT
#ifndef qsort
qsort() {}
#endif
#endif
