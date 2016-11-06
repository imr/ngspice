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

#ifdef USE_OMP
    /* free just once for all models */
    FREE(mod->BSIM4v5InstanceArray);
#endif

    while (mod) {
        BSIM4v5model *next_mod = mod->BSIM4v5nextModel;
        BSIM4v5instance *inst = mod->BSIM4v5instances;
        /** added to get rid of link list pSizeDependParamKnot **/
        struct bsim4v5SizeDependParam *p = mod->pSizeDependParamKnot;
        while (p) {
            struct bsim4v5SizeDependParam *next_p = p->pNext;
            FREE(p);
            p = next_p;
        }
        /** end of extra code **/
        while (inst) {
            BSIM4v5instance *next_inst = inst->BSIM4v5nextInstance;
            FREE(inst);
            inst = next_inst;
        }
        FREE(mod->BSIM4v5version);
        FREE(mod);
        mod = next_mod;
    }

    *inModel = NULL;
}
