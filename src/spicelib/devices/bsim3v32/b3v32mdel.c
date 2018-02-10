/**** BSIM3v3.2.4, Released by Xuemei Xi 12/21/2001 ****/

/**********
 * Copyright 2001 Regents of the University of California. All rights reserved.
 * File: b3mdel.c of BSIM3v3.2.4
 * Author: 1995 Min-Chie Jeng and Mansun Chan.
 * Author: 1997-1999 Weidong Liu.
 * Author: 2001  Xuemei Xi
 * Modified by Paolo Nenzi 2002
 **********/

#include "ngspice/ngspice.h"
#include "bsim3v32def.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
BSIM3v32mDelete(GENmodel *gen_model)
{
    BSIM3v32model *model = (BSIM3v32model *) gen_model;

#ifdef USE_OMP
    FREE(model->BSIM3v32InstanceArray);
#endif

    struct bsim3v32SizeDependParam *p =  model->pSizeDependParamKnot;
    while (p) {
        struct bsim3v32SizeDependParam *next_p = p->pNext;
        FREE(p);
        p = next_p;
    }

    FREE(model->BSIM3v32version);

    return OK;
}
