/**** BSIM4.6.2 Released by Wenwei Yang 07/31/2008 ****/

/**********
 * Copyright 2006 Regents of the University of California. All rights reserved.
 * File: b4mdel.c of BSIM4.6.2.
 * Author: 2000 Weidong Liu
 * Authors: 2001- Xuemei Xi, Mohan Dunga, Ali Niknejad, Chenming Hu.
 * Authors: 2006- Mohan Dunga, Ali Niknejad, Chenming Hu
 * Authors: 2007- Mohan Dunga, Wenwei Yang, Ali Niknejad, Chenming Hu
 * Project Director: Prof. Chenming Hu.
 **********/

#include "ngspice/ngspice.h"
#include "bsim4v6def.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
BSIM4v6mDelete(GENmodel *model)
{
    BSIM4v6model *mod = (BSIM4v6model*) model;

#ifdef USE_OMP
    /* free just once for all models */
    FREE(mod->BSIM4v6InstanceArray);
#endif

    struct bsim4v6SizeDependParam *p = mod->pSizeDependParamKnot;
    while (p) {
        struct bsim4v6SizeDependParam *next_p = p->pNext;
        FREE(p);
        p = next_p;
    }

    FREE(mod->BSIM4v6version);

    GENmodelFree(model);
    return OK;
}
