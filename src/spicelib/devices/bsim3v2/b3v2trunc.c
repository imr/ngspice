/* $Id$  */
/*
 $Log$
 Revision 1.1.1.1  2000-04-27 20:03:59  pnenzi
 Imported sources

 * Revision 3.2 1998/6/16  18:00:00  Weidong 
 * BSIM3v3.2 release
 *
*/
static char rcsid[] = "$Id$";

/*************************************/

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
register CKTcircuit *ckt;
double *timeStep;
{
register BSIM3V2model *model = (BSIM3V2model*)inModel;
register BSIM3V2instance *here;

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



