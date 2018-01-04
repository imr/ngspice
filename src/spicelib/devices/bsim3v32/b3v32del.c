/**** BSIM3v3.2.4, Released by Xuemei Xi 12/14/2001 ****/

/**********
 * Copyright 2001 Regents of the University of California. All rights reserved.
 * File: b3del.c of BSIM3v3.2.4
 * Author: 1995 Min-Chie Jeng and Mansun Chan
 * Author: 1997-1999 Weidong Liu.
 * Author: 2001 Xuemei Xi
 * Modified by Xuemei Xi, 10/05, 12/14, 2001.
 * Modified by Paolo Nenzi 2002
 **********/

#include "ngspice/ngspice.h"
#include "bsim3v32def.h"
#include "ngspice/sperror.h"
#include "ngspice/gendefs.h"
#include "ngspice/suffix.h"


int
BSIM3v32delete(GENmodel *model, IFuid name, GENinstance **kill)
{
    for (; model; model = model->GENnextModel) {
        GENinstance **prev = &(model->GENinstances);
        GENinstance *here = *prev;
        for (; here; here = *prev) {
            if (here->GENname == name || (kill && here == *kill)) {
                *prev = here->GENnextInstance;
                FREE(here);
                return OK;
            }
            prev = &(here->GENnextInstance);
        }
    }

    return E_NODEV;
}
