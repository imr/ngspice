/**** BSIM3v3.3.0, Released by Xuemei Xi 07/29/2005 ****/

/**********
 * Copyright 2004 Regents of the University of California. All rights reserved.
 * File: b3trunc.c of BSIM3v3.3.0
 * Author: 1995 Min-Chie Jeng and Mansun Chan. 
 * Author: 1997-1999 Weidong Liu.
 * Author: 2001  Xuemei Xi
 **********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "bsim3def.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
BSIM3trunc(
GENmodel *inModel,
CKTcircuit *ckt,
double *timeStep)
{
BSIM3model *model = (BSIM3model*)inModel;
BSIM3instance *here;

#ifdef STEPDEBUG
    double debugtemp;
#endif /* STEPDEBUG */

    for (; model != NULL; model = BSIM3nextModel(model))
    {    for (here = BSIM3instances(model); here != NULL;
	      here = BSIM3nextInstance(here))
	 {
#ifdef STEPDEBUG
            debugtemp = *timeStep;
#endif /* STEPDEBUG */
            CKTterr(here->BSIM3qb,ckt,timeStep);
            CKTterr(here->BSIM3qg,ckt,timeStep);
            CKTterr(here->BSIM3qd,ckt,timeStep);
#ifdef STEPDEBUG
            if(debugtemp != *timeStep)
	    {  printf("device %s reduces step from %g to %g\n",
                       here->BSIM3name,debugtemp,*timeStep);
            }
#endif /* STEPDEBUG */
        }
    }
    return(OK);
}
