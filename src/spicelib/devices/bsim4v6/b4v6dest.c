/**** BSIM4.6.2 Released by Wenwei Yang 07/31/2008 ****/

/**********
 * Copyright 2006 Regents of the University of California. All rights reserved.
 * File: b4dest.c of BSIM4.6.2.
 * Author: 2000 Weidong Liu
 * Authors: 2001- Xuemei Xi, Mohan Dunga, Ali Niknejad, Chenming Hu.
 * Authors: 2006- Mohan Dunga, Ali Niknejad, Chenming Hu
 * Authors: 2007- Mohan Dunga, Wenwei Yang, Ali Niknejad, Chenming Hu
 * Project Director: Prof. Chenming Hu.
 **********/

#include "ngspice/ngspice.h"
#include "bsim4v6def.h"
#include "ngspice/suffix.h"


void
BSIM4v6destroy(GENmodel **inModel)
{
    BSIM4v6model *mod = *(BSIM4v6model**) inModel;

    while (mod) {
        BSIM4v6model *next_mod = BSIM4v6nextModel(mod);
        BSIM4v6instance *inst = BSIM4v6instances(mod);
        while (inst) {
            BSIM4v6instance *next_inst = BSIM4v6nextInstance(inst);
            BSIM4v6delete(GENinstanceOf(inst));
            inst = next_inst;
        }
        BSIM4v6mDelete(GENmodelOf(mod));
        mod = next_mod;
    }

    *inModel = NULL;
}
