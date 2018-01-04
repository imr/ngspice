/**** BSIM4.6.2 Released by Wenwei Yang 07/31/2008 ****/

/**********
 * Copyright 2006 Regents of the University of California. All rights reserved.
 * File: b4del.c of BSIM4.6.2.
 * Author: 2000 Weidong Liu
 * Authors: 2001- Xuemei Xi, Mohan Dunga, Ali Niknejad, Chenming Hu.
 * Authors: 2006- Mohan Dunga, Ali Niknejad, Chenming Hu
 * Authors: 2007- Mohan Dunga, Wenwei Yang, Ali Niknejad, Chenming Hu
 * Project Director: Prof. Chenming Hu.
 **********/

#include "ngspice/ngspice.h"
#include "bsim4v6def.h"
#include "ngspice/sperror.h"
#include "ngspice/gendefs.h"
#include "ngspice/suffix.h"


int
BSIM4v6delete(
              GENmodel *inModel,
              IFuid name,
              GENinstance **inInst)
{
    BSIM4v6instance **fast = (BSIM4v6instance **) inInst;
    BSIM4v6model *model = (BSIM4v6model *) inModel;
    BSIM4v6instance **prev = NULL;
    BSIM4v6instance *here;

    for (; model; model = model->BSIM4v6nextModel) {
        prev = &(model->BSIM4v6instances);
        for (here = *prev; here; here = *prev) {
            if (here->BSIM4v6name == name || (fast && here == *fast)) {
                *prev = here->BSIM4v6nextInstance;
                FREE(here);
                return OK;
            }
            prev = &(here->BSIM4v6nextInstance);
        }
    }

    return E_NODEV;
}
