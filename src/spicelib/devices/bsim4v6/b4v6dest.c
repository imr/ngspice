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
BSIM4v6destroy(
    GENmodel **inModel)
{
    BSIM4v6model **model = (BSIM4v6model**)inModel;
    BSIM4v6instance *here;
    BSIM4v6instance *prev = NULL;
    BSIM4v6model *mod = *model;
    BSIM4v6model *oldmod = NULL;

    for (; mod ; mod = mod->BSIM4v6nextModel) {
    /** added to get rid of link list pSizeDependParamKnot **/      
        struct bsim4v6SizeDependParam *pParam, *pParamOld=NULL;

        pParam = mod->pSizeDependParamKnot;

        for (; pParam ; pParam = pParam->pNext) {
            FREE(pParamOld);
            pParamOld = pParam;
        }
        FREE(pParamOld);
        pParam = NULL;
     /** end of extra code **/

	if(oldmod)
            FREE(oldmod);  
        oldmod = mod;
        prev = NULL;
        for (here = mod->BSIM4v6instances; here; here = here->BSIM4v6nextInstance) {
            if(prev) FREE(prev);
            prev = here;
        }
        if(prev) FREE(prev);
    }
    if(oldmod) {
#ifdef USE_OMP
        /* free just once for all models */
        FREE(oldmod->BSIM4v6InstanceArray);
#endif
        FREE(oldmod);
    }
    *model = NULL;
    return;
}
