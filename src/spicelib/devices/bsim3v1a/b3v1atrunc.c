/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1995 Min-Chie Jeng and Mansun Chan.
Modified by Paolo Nenzi 2002
File: b3v1atrunc.c
**********/

#include "ngspice.h"
#include "cktdefs.h"
#include "bsim3v1adef.h"
#include "sperror.h"
#include "suffix.h"


int
BSIM3v1Atrunc(GENmodel *inModel, CKTcircuit *ckt, double *timeStep)
{
BSIM3v1Amodel *model = (BSIM3v1Amodel*)inModel;
BSIM3v1Ainstance *here;

#ifdef STEPDEBUG
    double debugtemp;
#endif /* STEPDEBUG */

    for (; model != NULL; model = model->BSIM3v1AnextModel)
    {    for (here = model->BSIM3v1Ainstances; here != NULL;
	      here = here->BSIM3v1AnextInstance)
	 {
	 
	    if (here->BSIM3v1Aowner != ARCHme)
                    continue;
 
#ifdef STEPDEBUG
            debugtemp = *timeStep;
#endif /* STEPDEBUG */
            CKTterr(here->BSIM3v1Aqb,ckt,timeStep);
            CKTterr(here->BSIM3v1Aqg,ckt,timeStep);
            CKTterr(here->BSIM3v1Aqd,ckt,timeStep);
#ifdef STEPDEBUG
            if(debugtemp != *timeStep)
	    {  printf("device %s reduces step from %g to %g\n",
                       here->BSIM3v1Aname,debugtemp,*timeStep);
            }
#endif /* STEPDEBUG */
        }
    }
    return(OK);
}



