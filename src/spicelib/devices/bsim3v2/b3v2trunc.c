/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1995 Min-Chie Jeng and Mansun Chan.
File: b3v2trunc.c
**********/

#include "ngspice.h"
#include <stdio.h>
#include <math.h>
#include "cktdefs.h"
#include "bsim3v2def.h"
#include "sperror.h"
#include "suffix.h"


int
BSIM3V2trunc(inModel,ckt,timeStep)
GENmodel *inModel;
CKTcircuit *ckt;
double *timeStep;
{
BSIM3V2model *model = (BSIM3V2model*)inModel;
BSIM3V2instance *here;

#ifdef STEPDEBUG
    double debugtemp;
#endif /* STEPDEBUG */

    for (; model != NULL; model = model->BSIM3V2nextModel)
    {    for (here = model->BSIM3V2instances; here != NULL;
	      here = here->BSIM3V2nextInstance)
	 {
         if (here->BSIM3V2owner != ARCHme) continue; 
#ifdef STEPDEBUG
            debugtemp = *timeStep;
#endif /* STEPDEBUG */
            CKTterr(here->BSIM3V2qb,ckt,timeStep);
            CKTterr(here->BSIM3V2qg,ckt,timeStep);
            CKTterr(here->BSIM3V2qd,ckt,timeStep);
#ifdef STEPDEBUG
            if(debugtemp != *timeStep)
	    {  printf("device %s reduces step from %g to %g\n",
                       here->BSIM3V2name,debugtemp,*timeStep);
            }
#endif /* STEPDEBUG */
        }
    }
    return(OK);
}



