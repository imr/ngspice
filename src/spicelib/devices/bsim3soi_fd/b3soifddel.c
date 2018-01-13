/**********
Copyright 1999 Regents of the University of California.  All rights reserved.
Author: 1998 Samuel Fung, Dennis Sinitsky and Stephen Tang
Modified by Paolo Nenzi 2002
File: b3soifddel.c          98/5/01
**********/

/*
 * Revision 2.1  99/9/27 Pin Su
 * BSIMFD2.1 release
 */

#include "ngspice/ngspice.h"
#include "b3soifddef.h"
#include "ngspice/sperror.h"
#include "ngspice/gendefs.h"
#include "ngspice/suffix.h"


int
B3SOIFDdelete(GENinstance *inst)
{
    GENinstanceFree(inst);
    return OK;
}
