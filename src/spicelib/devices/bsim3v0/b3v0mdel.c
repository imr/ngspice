/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1995 Min-Chie Jeng and Mansun Chan.
File: b3v0mdel.c
**********/

#include "ngspice/ngspice.h"
#include "bsim3v0def.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
BSIM3v0mDelete(GENmodel *gen_model)
{
    NG_IGNORE(gen_model);
    return OK;
}
