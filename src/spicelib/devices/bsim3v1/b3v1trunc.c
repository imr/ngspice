/**********
 * Copyright 1990 Regents of the University of California. All rights reserved.
 * File: b3v1ld.c
 * Author: 1995 Min-Chie Jeng and Mansun Chan. 
 * Modified by Paolo Nenzi 2002
 **********/
 
/* 
 * Release Notes: 
 * BSIM3v3.1,   Released by yuhua  96/12/08
 */

#include "ngspice.h"
#include "cktdefs.h"
#include "bsim3v1def.h"
#include "sperror.h"
#include "suffix.h"


int
BSIM3v1trunc(GENmodel *inModel, CKTcircuit *ckt, double *timeStep)
{
BSIM3v1model *model = (BSIM3v1model*)inModel;
BSIM3v1instance *here;

#ifdef STEPDEBUG
    double debugtemp;
#endif /* STEPDEBUG */

    for (; model != NULL; model = model->BSIM3v1nextModel)
    {    for (here = model->BSIM3v1instances; here != NULL;
	      here = here->BSIM3v1nextInstance)
	 {
	 
            if (here->BSIM3v1owner != ARCHme)
                    continue;

#ifdef STEPDEBUG
            debugtemp = *timeStep;
#endif /* STEPDEBUG */
            CKTterr(here->BSIM3v1qb,ckt,timeStep);
            CKTterr(here->BSIM3v1qg,ckt,timeStep);
            CKTterr(here->BSIM3v1qd,ckt,timeStep);
#ifdef STEPDEBUG
            if(debugtemp != *timeStep)
	    {  printf("device %s reduces step from %g to %g\n",
                       here->BSIM3v1name,debugtemp,*timeStep);
            }
#endif /* STEPDEBUG */
        }
    }
    return(OK);
}



