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
        BSIM4v5model *next_mod = mod->BSIM4v5nextModel;
        BSIM4v5instance *inst = mod->BSIM4v5instances;
        while (inst) {
            BSIM4v5instance *next_inst = inst->BSIM4v5nextInstance;
            FREE(inst);
            inst = next_inst;
        }

        FREE(mod);
        mod = next_mod;
    }

    *inModel = NULL;
}
