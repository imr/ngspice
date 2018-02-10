/**** BSIM4.7.0 Released by Darsen Lu 04/08/2011 ****/

/**********
 * Copyright 2006 Regents of the University of California. All rights reserved.
 * File: b4dest.c of BSIM4.7.0.
 * Author: 2000 Weidong Liu
 * Authors: 2001- Xuemei Xi, Mohan Dunga, Ali Niknejad, Chenming Hu.
 * Authors: 2006- Mohan Dunga, Ali Niknejad, Chenming Hu
 * Authors: 2007- Mohan Dunga, Wenwei Yang, Ali Niknejad, Chenming Hu
 * Project Director: Prof. Chenming Hu.
 **********/

#include "ngspice/ngspice.h"
#include "bsim4v7def.h"
#include "ngspice/suffix.h"


void
BSIM4v7destroy(GENmodel **inModel)
{
    BSIM4v7model *mod = *(BSIM4v7model**) inModel;

    while (mod) {
        BSIM4v7model *next_mod = mod->BSIM4v7nextModel;
        BSIM4v7instance *inst = mod->BSIM4v7instances;
        while (inst) {
            BSIM4v7instance *next_inst = inst->BSIM4v7nextInstance;
            FREE(inst);
            inst = next_inst;
        }

        FREE(mod);
        mod = next_mod;
    }

    *inModel = NULL;
}
