/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1998 Samuel Fung, Dennis Sinitsky and Stephen Tang
File: b3soitrunc.c          98/5/01
Modified by Paolo Nenzi 2002
**********/


#include "ngspice.h"
#include "cktdefs.h"
#include "b3soidef.h"
#include "sperror.h"
#include "suffix.h"


int
B3SOItrunc(GENmodel *inModel, CKTcircuit *ckt, double *timeStep)
{
B3SOImodel *model = (B3SOImodel*)inModel;
B3SOIinstance *here;

#ifdef STEPDEBUG
    double debugtemp;
#endif /* STEPDEBUG */

    for (; model != NULL; model = model->B3SOInextModel)
    {    for (here = model->B3SOIinstances; here != NULL;
	      here = here->B3SOInextInstance)
	 {
            if (here->B3SOIowner != ARCHme)
                    continue;

#ifdef STEPDEBUG
            debugtemp = *timeStep;
#endif /* STEPDEBUG */
            CKTterr(here->B3SOIqb,ckt,timeStep);
            CKTterr(here->B3SOIqg,ckt,timeStep);
            CKTterr(here->B3SOIqd,ckt,timeStep);
#ifdef STEPDEBUG
            if(debugtemp != *timeStep)
	    {  printf("device %s reduces step from %g to %g\n",
                       here->B3SOIname,debugtemp,*timeStep);
            }
#endif /* STEPDEBUG */
        }
    }
    return(OK);
}



