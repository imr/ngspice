/**** BSIM4.5.0 Released by Xuemei (Jane) Xi 07/29/2005 ****/

/**********
 * Copyright 2005 Regents of the University of California. All rights reserved.
 * File: b4dest.c of BSIM4.5.0.
 * Author: 2000 Weidong Liu
 * Authors: 2001- Xuemei Xi, Mohan Dunga, Ali Niknejad, Chenming Hu.
 * Project Director: Prof. Chenming Hu.
 **********/

#include "ngspice/ngspice.h"
#include "bsim4v5def.h"
#include "ngspice/suffix.h"


void
BSIM4v5destroy(GENmodel **inModel)
{
    BSIM4v5model *mod = *(BSIM4v5model**) inModel;

    while (mod) {
        BSIM4v5model *next_mod = BSIM4v5nextModel(mod);
        BSIM4v5instance *inst = BSIM4v5instances(mod);
        while (inst) {
            BSIM4v5instance *next_inst = BSIM4v5nextInstance(inst);
            BSIM4v5delete(GENinstanceOf(inst));
            inst = next_inst;
        }
        BSIM4v5mDelete(GENmodelOf(mod));
        mod = next_mod;
    }

    *inModel = NULL;
}
