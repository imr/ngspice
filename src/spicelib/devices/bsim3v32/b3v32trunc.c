/**** BSIM3v3.2.4, Released by Xuemei Xi 12/21/2001 ****/

/**********
 * Copyright 2001 Regents of the University of California. All rights reserved.
 * File: b3trunc.c of BSIM3v3.2.4
 * Author: 1995 Min-Chie Jeng and Mansun Chan.
 * Author: 1997-1999 Weidong Liu.
 * Author: 2001  Xuemei Xi
 * Modified by Poalo Nenzi 2002
 **********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "bsim3v32def.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
BSIM3v32trunc (GENmodel *inModel, CKTcircuit *ckt, double *timeStep)
{
BSIM3v32model *model = (BSIM3v32model*)inModel;
BSIM3v32instance *here;

#ifdef STEPDEBUG
    double debugtemp;
#endif /* STEPDEBUG */

    for (; model != NULL; model = BSIM3v32nextModel(model))
    {    for (here = BSIM3v32instances(model); here != NULL;
              here = BSIM3v32nextInstance(here))
         {
#ifdef STEPDEBUG
            debugtemp = *timeStep;
#endif /* STEPDEBUG */
            CKTterr(here->BSIM3v32qb,ckt,timeStep);
            CKTterr(here->BSIM3v32qg,ckt,timeStep);
            CKTterr(here->BSIM3v32qd,ckt,timeStep);
#ifdef STEPDEBUG
            if(debugtemp != *timeStep)
            {  printf("device %s reduces step from %g to %g\n",
                       here->BSIM3v32name,debugtemp,*timeStep);
            }
#endif /* STEPDEBUG */
        }
    }
    return(OK);
}
