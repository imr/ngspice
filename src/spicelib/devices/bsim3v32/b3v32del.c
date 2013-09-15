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
BSIM3v32delete (GENmodel *inModel, IFuid name, GENinstance **inInst)
{
BSIM3v32instance **fast = (BSIM3v32instance**)inInst;
BSIM3v32model *model = (BSIM3v32model*)inModel;
BSIM3v32instance **prev = NULL;
BSIM3v32instance *here;

    for (; model ; model = model->BSIM3v32nextModel)
    {    prev = &(model->BSIM3v32instances);
         for (here = *prev; here ; here = *prev)
         {    if (here->BSIM3v32name == name || (fast && here==*fast))
              {   *prev= here->BSIM3v32nextInstance;
                  FREE(here);
                  return(OK);
              }
              prev = &(here->BSIM3v32nextInstance);
         }
    }
    return(E_NODEV);
}
