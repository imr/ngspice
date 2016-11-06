/**** BSIM4.8.0 Released by Navid Paydavosi 11/01/2013 ****/

/**********
 * Copyright 2006 Regents of the University of California. All rights reserved.
 * File: b4dest.c of BSIM4.8.0.
 * Author: 2000 Weidong Liu
 * Authors: 2001- Xuemei Xi, Mohan Dunga, Ali Niknejad, Chenming Hu.
 * Authors: 2006- Mohan Dunga, Ali Niknejad, Chenming Hu
 * Authors: 2007- Mohan Dunga, Wenwei Yang, Ali Niknejad, Chenming Hu
 * Project Director: Prof. Chenming Hu.
 **********/

#include "ngspice/ngspice.h"
#include "bsim4def.h"
#include "ngspice/suffix.h"


void
BSIM4destroy(GENmodel **inModel)
{
    BSIM4model *mod = *(BSIM4model**) inModel;

    while (mod) {
        BSIM4model *next_mod = mod->BSIM4nextModel;
        BSIM4instance *inst = mod->BSIM4instances;
        /** added to get rid of link list pSizeDependParamKnot **/
        struct bsim4SizeDependParam *p = mod->pSizeDependParamKnot;
        while (p) {
            struct bsim4SizeDependParam *next_p = p->pNext;
            FREE(p);
            p = next_p;
        }
        /** end of extra code **/
        while (inst) {
            BSIM4instance *next_inst = inst->BSIM4nextInstance;
            FREE(inst);
            inst = next_inst;
        }
#ifdef USE_OMP
        FREE(mod->BSIM4InstanceArray);
#endif
        FREE(mod->BSIM4version);
        FREE(mod);
        mod = next_mod;
    }

    *inModel = NULL;
}
