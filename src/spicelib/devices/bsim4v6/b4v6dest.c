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
        BSIM4v6model *next_mod = mod->BSIM4v6nextModel;
        BSIM4v6instance *inst = mod->BSIM4v6instances;
        /** added to get rid of link list pSizeDependParamKnot **/
        struct bsim4v6SizeDependParam *p = mod->pSizeDependParamKnot;
        while (p) {
            struct bsim4v6SizeDependParam *next_p = p->pNext;
            FREE(p);
            p = next_p;
        }
        /** end of extra code **/
        while (inst) {
            BSIM4v6instance *next_inst = inst->BSIM4v6nextInstance;
            FREE(inst);
            inst = next_inst;
        }

#ifdef USE_OMP
        FREE(mod->BSIM4v6InstanceArray);
#endif

        FREE(mod->BSIM4v6version);
        FREE(mod);
        mod = next_mod;
    }

    *inModel = NULL;
}
