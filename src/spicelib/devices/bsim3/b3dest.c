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
BSIM3destroy(GENmodel **inModel)
{
    BSIM3model *mod = *(BSIM3model**) inModel;

    while (mod) {
        BSIM3model *next_mod = mod->BSIM3nextModel;
        BSIM3instance *inst = mod->BSIM3instances;

        /** added to get rid of link list pSizeDependParamKnot **/
        struct bsim3SizeDependParam *p = mod->pSizeDependParamKnot;
        while (p) {
            struct bsim3SizeDependParam *next_p = p->pNext;
            FREE(p);
            p = next_p;
        }
        /** end of extra code **/

        while (inst) {
            BSIM3instance *next_inst = inst->BSIM3nextInstance;
            FREE(inst);
            inst = next_inst;
        }

#ifdef USE_OMP
        FREE(mod->BSIM3InstanceArray);
#endif

        /* mod->BSIM3modName to be freed in INPtabEnd() */
        FREE(mod->BSIM3version);
        FREE(mod);
        mod = next_mod;
    }

    *inModel = NULL;
}
