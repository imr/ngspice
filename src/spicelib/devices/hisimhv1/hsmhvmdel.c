/***********************************************************************

 HiSIM (Hiroshima University STARC IGFET Model)
 Copyright (C) 2012 Hiroshima University & STARC

 MODEL NAME : HiSIM_HV
 ( VERSION : 1  SUBVERSION : 2  REVISION : 4 )
 Model Parameter VERSION : 1.23
 FILE : hsmhvmdel.c

 DATE : 2013.04.30

 released by
                Hiroshima University &
                Semiconductor Technology Academic Research Center (STARC)
***********************************************************************/

#include "ngspice/ngspice.h"
#include "hsmhvdef.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
HSMHVmDelete(GENmodel **models, IFuid modname, GENmodel *kill)
{
    GENinstance *here;
    GENmodel **prev = models;
    GENmodel *model = *prev;

    for (; model; model = model->GENnextModel) {
        if (model->GENmodName == modname || (kill && model == kill))
            goto delgot;
        prev = &(model->GENnextModel);
    }

    return E_NOMOD;

 delgot:
    *prev = model->GENnextModel;
    for (here = model->GENinstances; here;) {
        GENinstance *next_instance = here->GENnextInstance;
        FREE(here);
        here = next_instance;
    }
    FREE(model);
    return OK;
}
