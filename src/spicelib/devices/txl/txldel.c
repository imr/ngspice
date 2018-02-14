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
TXLdelete(GENinstance *gen_inst)
{
    NG_IGNORE(gen_inst);
    return OK;
}
