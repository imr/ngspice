/**********
Copyright 1992 Regents of the University of California.  All rights reserved.
Author: 1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
**********/

/*
 * This routine deletes all NBJTs from the circuit and frees all storage they
 * were using.  The current implementation has memory leaks.
 */

#include "ngspice/ngspice.h"
#include "nbjtdefs.h"
#include "ngspice/suffix.h"


void
NBJTdestroy(void)
{
}
