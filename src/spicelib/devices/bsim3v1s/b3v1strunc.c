/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1995 Min-Chie Jeng and Mansun Chan.
Modified by Paolo Nenzi 2002
File: b3v1strunc.c
**********/

#include "ngspice.h"
#include "cktdefs.h"
#include "bsim3v1sdef.h"
#include "sperror.h"
#include "suffix.h"


int
BSIM3v1Strunc(GENmodel *inModel, CKTcircuit *ckt, double *timeStep)
{
BSIM3v1Smodel *model = (BSIM3v1Smodel*)inModel;
BSIM3v1Sinstance *here;

#ifdef STEPDEBUG
    double debugtemp;
#endif /* STEPDEBUG */

    for (; model != NULL; model = model->BSIM3v1SnextModel)
    {    for (here = model->BSIM3v1Sinstances; here != NULL;
	      here = here->BSIM3v1SnextInstance)
	 {
        
	 if (here->BSIM3v1Sowner != ARCHme) 
	         continue;

#ifdef STEPDEBUG
            debugtemp = *timeStep;
#endif /* STEPDEBUG */
            CKTterr(here->BSIM3v1Sqb,ckt,timeStep);
            CKTterr(here->BSIM3v1Sqg,ckt,timeStep);
            CKTterr(here->BSIM3v1Sqd,ckt,timeStep);
#ifdef STEPDEBUG
            if(debugtemp != *timeStep)
	    {  printf("device %s reduces step from %g to %g\n",
                       here->BSIM3v1Sname,debugtemp,*timeStep);
            }
#endif /* STEPDEBUG */

        }
     }
    return(OK);
}
