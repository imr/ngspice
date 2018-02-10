/**** BSIM4.5.0 Released by Xuemei (Jane) Xi 07/29/2005 ****/

/**********
 * Copyright 2005 Regents of the University of California. All rights reserved.
 * File: b4mdel.c of BSIM4.5.0.
 * Author: 2000 Weidong Liu
 * Authors: 2001- Xuemei Xi, Mohan Dunga, Ali Niknejad, Chenming Hu.
 * Project Director: Prof. Chenming Hu.
 **********/

#include "ngspice/ngspice.h"
#include "bsim4v5def.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
BSIM4v5mDelete(GENmodel *gen_model)
{
    BSIM4v5model *model = (BSIM4v5model*) gen_model;

#ifdef USE_OMP
    FREE(model->BSIM4v5InstanceArray);
#endif

    struct bsim4v5SizeDependParam *p = model->pSizeDependParamKnot;
    while (p) {
        struct bsim4v5SizeDependParam *next_p = p->pNext;
        FREE(p);
        p = next_p;
    }

    FREE(model->BSIM4v5version);

    return OK;
}
