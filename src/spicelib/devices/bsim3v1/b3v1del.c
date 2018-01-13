/**********
 * Copyright 1990 Regents of the University of California. All rights reserved.
 * File: b3v1del.c
 * Author: 1995 Min-Chie Jeng and Mansun Chan.
 * Modified by Paolo Nenzi 2002
 **********/

/*
 * Release Notes:
 * BSIM3v3.1,   Released by yuhua  96/12/08
 */

#include "ngspice/ngspice.h"
#include "bsim3v1def.h"
#include "ngspice/sperror.h"
#include "ngspice/gendefs.h"
#include "ngspice/suffix.h"


int
BSIM3v1delete(GENinstance *inst)
{
    GENinstanceFree(inst);
    return OK;
}
