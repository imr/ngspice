/**** BSIM3v3.2.4, Released by Xuemei Xi 12/14/2001 ****/

/**********
 * Copyright 2001 Regents of the University of California. All rights reserved.
 * File: b3dest.c of BSIM3v3.2.4
 * Author: 1995 Min-Chie Jeng and Mansun Chan.
 * Author: 1997-1999 Weidong Liu.
 * Author: 2001  Xuemei Xi
 * Modified by Paolo Nenzi 2002
 **********/

#include "ngspice/ngspice.h"
#include "bsim3v32def.h"
#include "ngspice/suffix.h"


void
BSIM3v32destroy (GENmodel **inModel)
{
    BSIM3v32model *mod = *(BSIM3v32model**) inModel;

#ifdef USE_OMP
    /* free just once for all models */
    FREE(mod->BSIM3v32InstanceArray);
#endif

    while (mod) {
        BSIM3v32model *next_mod = mod->BSIM3v32nextModel;
        BSIM3v32instance *inst = mod->BSIM3v32instances;
        /** added to get rid of link list pSizeDependParamKnot **/
        struct bsim3v32SizeDependParam *p =  mod->pSizeDependParamKnot;
        while (p) {
            struct bsim3v32SizeDependParam *next_p = p->pNext;
            FREE(p);
            p = next_p;
        }
        /** end of extra code **/
        while (inst) {
            BSIM3v32instance *next_inst = inst->BSIM3v32nextInstance;
            FREE(inst);
            inst = next_inst;
        }
        FREE(mod->BSIM3v32version);
        FREE(mod);
        mod = next_mod;
    }

    *inModel = NULL;
}
