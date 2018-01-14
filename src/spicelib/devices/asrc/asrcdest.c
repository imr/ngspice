/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1987 Kanwar Jit Singh
**********/

#include "ngspice/ngspice.h"
#include "asrcdefs.h"
#include "ngspice/suffix.h"


void
ASRCdestroy(void)
{
    FREE(asrc_vals);
    FREE(asrc_derivs);
    asrc_nvals = 0;
}
