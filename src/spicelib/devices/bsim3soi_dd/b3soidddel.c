/**********
Copyright 1999 Regents of the University of California.  All rights reserved.
Author: 1998 Samuel Fung, Dennis Sinitsky and Stephen Tang
File: b3soidddel.c          98/5/01
Modified by Paolo Nenzi 2002
**********/

/*
 * Revision 2.1  99/9/27 Pin Su
 * BSIMDD2.1 release
 */

#include "ngspice/ngspice.h"
#include "b3soidddef.h"
#include "ngspice/sperror.h"
#include "ngspice/gendefs.h"
#include "ngspice/suffix.h"


int
B3SOIDDdelete(GENinstance *inst)
{
    GENinstanceFree(inst);
    return OK;
}
