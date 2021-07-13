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


int
CPLmDelete(GENmodel *gen_model)
{
    CPLmodel *model = (CPLmodel *)gen_model;

    FREE(model->Rm);
    FREE(model->Lm);
    FREE(model->Gm);
    FREE(model->Cm);
    return OK;
}
