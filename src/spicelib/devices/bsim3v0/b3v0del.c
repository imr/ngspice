/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1995 Min-Chie Jeng and Mansun Chan.
File: b3v0del.c
**********/

#include "ngspice/ngspice.h"
#include "bsim3v0def.h"
#include "ngspice/sperror.h"
#include "ngspice/gendefs.h"
#include "ngspice/suffix.h"


int
BSIM3v0delete(GENinstance *gen_inst)
{
    NG_IGNORE(gen_inst);
    return OK;
}
