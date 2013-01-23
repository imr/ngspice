/**********
Copyright 1991 Regents of the University of California.  All rights reserved.
Author:	1991 David A. Gates, U. C. Berkeley CAD Group
**********/

/* Member of CIDER device simulator
 * Version: 1b1
 */

/**********
 Mesh Definitions and Declarations.
**********/
#ifndef ngspice_MESHEXT_H
#define ngspice_MESHEXT_H

#include "ngspice/meshdefs.h"
#include "ngspice/gendev.h"


extern double *MESHmkArray( MESHcoord *, int );
extern void MESHiBounds( MESHcoord *, int *, int * );
extern void MESHlBounds( MESHcoord *, double *, double * );
extern int MESHlocate( MESHcoord *, double );
extern int MESHcheck( char, MESHcard * );
extern int MESHsetup( char, MESHcard *, MESHcoord **, int * );

#endif
