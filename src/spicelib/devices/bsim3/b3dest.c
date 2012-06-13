/**** BSIM3v3.3.0, Released by Xuemei Xi 07/29/2005 ****/

/**********
 * Copyright 2004 Regents of the University of California. All rights reserved.
 * File: b3dest.c of BSIM3v3.3.0
 * Author: 1995 Min-Chie Jeng and Mansun Chan.
 * Author: 1997-1999 Weidong Liu.
 * Author: 2001  Xuemei Xi
 **********/

#include "ngspice/ngspice.h"
#include "bsim3def.h"
#include "ngspice/suffix.h"

void
BSIM3destroy(
    GENmodel **inModel)
{
    BSIM3model **model = (BSIM3model**)inModel;
    BSIM3instance *here;
    BSIM3instance *prev = NULL;
    BSIM3model *mod = *model;
    BSIM3model *oldmod = NULL;

    for (; mod ; mod = mod->BSIM3nextModel) {
    /** added to get rid of link list pSizeDependParamKnot **/      
        struct bsim3SizeDependParam *pParam, *pParamOld=NULL;

        pParam = mod->pSizeDependParamKnot;

        for (; pParam ; pParam = pParam->pNext) {
            FREE(pParamOld);
            pParamOld = pParam;
        }
        FREE(pParamOld);
        pParam = NULL;
     /** end of extra code **/
        if(oldmod) FREE(oldmod);
        oldmod = mod;
        prev = NULL;
        for (here = mod->BSIM3instances; here; here = here->BSIM3nextInstance) {
            if(prev) FREE(prev);
            prev = here;
        }
        if(prev) FREE(prev);
    }
    if(oldmod) {
#ifdef USE_OMP
        /* free just once for all models */
        FREE(oldmod->BSIM3InstanceArray);
#endif
        FREE(oldmod);
    }
    *model = NULL;
    return;
}



