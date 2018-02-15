/**********
Copyright 1992 Regents of the University of California.  All rights
reserved.
Author: 1992 Charles Hough
**********/

#include "ngspice/ngspice.h"
#include "txldefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
TXLmDelete(GENmodel *gen_model)
{
    NG_IGNORE(gen_model);
    return OK;
}
