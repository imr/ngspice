/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1995 Min-Chie Jeng and Mansun Chan.
File: b3v1trunc.c
**********/

#include "ngspice.h"
#include <stdio.h>
#include <math.h>
#include "cktdefs.h"
#include "bsim3v1def.h"
#include "sperror.h"
#include "suffix.h"


int
BSIM3V1trunc(inModel,ckt,timeStep)
GENmodel *inModel;
CKTcircuit *ckt;
double *timeStep;
{
BSIM3V1model *model = (BSIM3V1model*)inModel;
BSIM3V1instance *here;

#ifdef STEPDEBUG
    double debugtemp;
#endif /* STEPDEBUG */

    for (; model != NULL; model = model->BSIM3V1nextModel)
    {    for (here = model->BSIM3V1instances; here != NULL;
	      here = here->BSIM3V1nextInstance)
	 {
         if (here->BSIM3V1owner != ARCHme) continue;
#ifdef STEPDEBUG
            debugtemp = *timeStep;
#endif /* STEPDEBUG */
            CKTterr(here->BSIM3V1qb,ckt,timeStep);
            CKTterr(here->BSIM3V1qg,ckt,timeStep);
            CKTterr(here->BSIM3V1qd,ckt,timeStep);
#ifdef STEPDEBUG
            if(debugtemp != *timeStep)
	    {  printf("device %s reduces step from %g to %g\n",
                       here->BSIM3V1name,debugtemp,*timeStep);
            }
#endif /* STEPDEBUG */

        }
     }
    return(OK);
}
