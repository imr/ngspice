/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 S. Hwang
**********/

/*
  Imported into hfet2 model: Paolo Nenzi 2001
*/

#include "ngspice/ngspice.h"
#include "hfet2defs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
HFET2mDelete(GENmodel *gen_model)
{
    NG_IGNORE(gen_model);
    return OK;
}
