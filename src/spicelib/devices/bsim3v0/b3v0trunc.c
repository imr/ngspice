/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1995 Min-Chie Jeng and Mansun Chan.
File: b3v0trunc.c
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "bsim3v0def.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
BSIM3v0trunc(GENmodel *inModel, CKTcircuit *ckt, double *timeStep)
{
BSIM3v0model *model = (BSIM3v0model*)inModel;
BSIM3v0instance *here;

#ifdef STEPDEBUG
    double debugtemp;
#endif /* STEPDEBUG */

    for (; model != NULL; model = BSIM3v0nextModel(model))
    {    for (here = BSIM3v0instances(model); here != NULL;
	      here = BSIM3v0nextInstance(here))
	 {

#ifdef STEPDEBUG
            debugtemp = *timeStep;
#endif /* STEPDEBUG */
            CKTterr(here->BSIM3v0qb,ckt,timeStep);
            CKTterr(here->BSIM3v0qg,ckt,timeStep);
            CKTterr(here->BSIM3v0qd,ckt,timeStep);
#ifdef STEPDEBUG
            if(debugtemp != *timeStep)
	    {  printf("device %s reduces step from %g to %g\n",
                       here->BSIM3v0name,debugtemp,*timeStep);
            }
#endif /* STEPDEBUG */
        }
    }
    return(OK);
}
