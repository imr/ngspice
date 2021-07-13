/**********
Copyright 2021 The ngspice team  All rights
reserved.
Author: 2021 Holger Vogt
3-clause BSD license
**********/

#include "ngspice/ngspice.h"
#include "cpldefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"
#include "cplhash.h"

int
CPLDelete(GENinstance *gen_inst)
{
    NG_IGNORE(gen_inst);

//    mem_delete();
    return OK;
}
