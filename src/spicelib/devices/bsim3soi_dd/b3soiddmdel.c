/**********
Copyright 1999 Regents of the University of California.  All rights reserved.
Author: Weidong Liu and Pin Su         Feb 1999
Author: 1998 Samuel Fung, Dennis Sinitsky and Stephen Tang
File: b3soiddmdel.c          98/5/01
Modified by Paolo Nenzi 2002
**********/

/*
 * Revision 2.1  99/9/27 Pin Su
 * BSIMDD2.1 release
 */

#include "ngspice/ngspice.h"
#include "b3soidddef.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
B3SOIDDmDelete(GENmodel *gen_model)
{
    NG_IGNORE(gen_model);
    return OK;
}
