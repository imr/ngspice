/**** BSIM3v3.3.0, Released by Xuemei Xi 07/29/2005 ****/

/**********
 * Copyright 2004 Regents of the University of California. All rights reserved.
 * File: b3mdel.c of BSIM3v3.3.0
 * Author: 1995 Min-Chie Jeng and Mansun Chan.
 * Author: 1997-1999 Weidong Liu.
 * Author: 2001  Xuemei Xi
 **********/

#include "ngspice/ngspice.h"
#include "bsim3def.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
BSIM3mDelete(GENmodel *gen_model)
{
    BSIM3model *model = (BSIM3model*) gen_model;

#ifdef USE_OMP
    FREE(model->BSIM3InstanceArray);
#endif

    struct bsim3SizeDependParam *p = model->pSizeDependParamKnot;
    while (p) {
        struct bsim3SizeDependParam *next_p = p->pNext;
        FREE(p);
        p = next_p;
    }

    /* model->BSIM3modName to be freed in INPtabEnd() */
    FREE(model->BSIM3version);

    return OK;
}
