/**** BSIM4.7.0 Released by Darsen Lu 04/08/2011 ****/

/**********
 * Copyright 2006 Regents of the University of California. All rights reserved.
 * File: b4mdel.c of BSIM4.7.0.
 * Author: 2000 Weidong Liu
 * Authors: 2001- Xuemei Xi, Mohan Dunga, Ali Niknejad, Chenming Hu.
 * Authors: 2006- Mohan Dunga, Ali Niknejad, Chenming Hu
 * Authors: 2007- Mohan Dunga, Wenwei Yang, Ali Niknejad, Chenming Hu
 * Project Director: Prof. Chenming Hu.
 **********/

#include "ngspice/ngspice.h"
#include "bsim4v7def.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
BSIM4v7mDelete(GENmodel *model)
{
    BSIM4v7model *mod = (BSIM4v7model*) model;

#ifdef USE_OMP
    /* FIXME free just once for all models */
    FREE(mod->BSIM4v7InstanceArray);
#endif

    struct bsim4SizeDependParam *p = mod->pSizeDependParamKnot;
    while (p) {
        struct bsim4SizeDependParam *next_p = p->pNext;
        FREE(p);
        p = next_p;
    }

    FREE(mod->BSIM4v7version);

    GENmodelFree(model);
    return OK;
}
