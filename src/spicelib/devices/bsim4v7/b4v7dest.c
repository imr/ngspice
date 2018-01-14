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
        BSIM4v7model *next_mod = BSIM4v7nextModel(mod);
        BSIM4v7instance *inst = BSIM4v7instances(mod);
        /** added to get rid of link list pSizeDependParamKnot **/
        struct bsim4SizeDependParam *p = mod->pSizeDependParamKnot;
        while (p) {
            struct bsim4SizeDependParam *next_p = p->pNext;
            FREE(p);
            p = next_p;
        }
        /** end of extra code **/
        while (inst) {
            BSIM4v7instance *next_inst = BSIM4v7nextInstance(inst);
            GENinstanceFree(GENinstanceOf(inst));
            inst = next_inst;
        }

#ifdef USE_OMP
        FREE(mod->BSIM4v7InstanceArray);
#endif

        /* mod->BSIM4v7modName to be freed in INPtabEnd() */
        FREE(mod->BSIM4v7version);
        GENmodelFree(GENmodelOf(mod));
        mod = next_mod;
    }

    *inModel = NULL;
}
