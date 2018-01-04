/**** BSIM4.5.0 Released by Xuemei (Jane) Xi 07/29/2005 ****/

/**********
 * Copyright 2005 Regents of the University of California. All rights reserved.
 * File: b4del.c of BSIM4.5.0.
 * Author: 2000 Weidong Liu
 * Authors: 2001- Xuemei Xi, Mohan Dunga, Ali Niknejad, Chenming Hu.
 * Project Director: Prof. Chenming Hu.
 **********/

#include "ngspice/ngspice.h"
#include "bsim4v5def.h"
#include "ngspice/sperror.h"
#include "ngspice/gendefs.h"
#include "ngspice/suffix.h"


int
BSIM4v5delete(GENmodel *inModel, IFuid name, GENinstance **inInst)
{
    BSIM4v5instance **fast = (BSIM4v5instance **) inInst;
    BSIM4v5model *model = (BSIM4v5model *) inModel;
    BSIM4v5instance **prev = NULL;
    BSIM4v5instance *here;

    for (; model; model = model->BSIM4v5nextModel) {
        prev = &(model->BSIM4v5instances);
        for (here = *prev; here; here = *prev) {
            if (here->BSIM4v5name == name || (fast && here == *fast)) {
                *prev = here->BSIM4v5nextInstance;
                FREE(here);
                return OK;
            }
            prev = &(here->BSIM4v5nextInstance);
        }
    }

    return E_NODEV;
}
