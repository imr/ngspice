/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1998 Samuel Fung, Dennis Sinitsky and Stephen Tang
File: b3soipddel.c          98/5/01
Modified by Paolo Nenzi 2002
**********/

/*
 * Revision 2.2.3  02/3/5  Pin Su
 * BSIMPD2.2.3 release
 */

#include "ngspice/ngspice.h"
#include "b3soipddef.h"
#include "ngspice/sperror.h"
#include "ngspice/gendefs.h"
#include "ngspice/suffix.h"


int
B3SOIPDdelete(GENinstance *inst)
{
    GENinstanceFree(inst);
    return OK;
}
