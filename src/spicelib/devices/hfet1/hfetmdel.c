/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 S. Hwang
**********/

/*
  Imported into hfeta model: Paolo Nenzi 2001
*/

#include "ngspice/ngspice.h"
#include "hfetdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
HFETAmDelete(GENmodel *model)
{
    GENmodelFree(model);
    return OK;
}
