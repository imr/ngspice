/**********
Copyright 1999 Regents of the University of California.  All rights reserved.
Author: 1998 Samuel Fung, Dennis Sinitsky and Stephen Tang
File: b3soifdtrunc.c          98/5/01
**********/


#include "ngspice.h"
#include <stdio.h>
#include <math.h>
#include "cktdefs.h"
#include "b3soifddef.h"
#include "sperror.h"
#include "suffix.h"


int
B3SOIFDtrunc(inModel,ckt,timeStep)
GENmodel *inModel;
 CKTcircuit *ckt;
double *timeStep;
{
 B3SOIFDmodel *model = (B3SOIFDmodel*)inModel;
 B3SOIFDinstance *here;

#ifdef STEPDEBUG
    double debugtemp;
#endif /* STEPDEBUG */

    for (; model != NULL; model = model->B3SOIFDnextModel)
    {    for (here = model->B3SOIFDinstances; here != NULL;
	      here = here->B3SOIFDnextInstance)
	 {
#ifdef STEPDEBUG
            debugtemp = *timeStep;
#endif /* STEPDEBUG */
            CKTterr(here->B3SOIFDqb,ckt,timeStep);
            CKTterr(here->B3SOIFDqg,ckt,timeStep);
            CKTterr(here->B3SOIFDqd,ckt,timeStep);
#ifdef STEPDEBUG
            if(debugtemp != *timeStep)
	    {  printf("device %s reduces step from %g to %g\n",
                       here->B3SOIFDname,debugtemp,*timeStep);
            }
#endif /* STEPDEBUG */
        }
    }
    return(OK);
}



