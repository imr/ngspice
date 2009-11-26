/**** BSIM4.3.0 Released by Xuemei(Jane) Xi 05/09/2003 ****/

/**********
 * Copyright 2003 Regents of the University of California. All rights reserved.
 * File: b4v3check.c of BSIM4.3.0.
 * Author: 2000 Weidong Liu
 * Authors: 2001- Xuemei Xi, Jin He, Kanyu Cao, Mohan Dunga, Mansun Chan, Ali Niknejad, Chenming Hu.
 * Project Director: Prof. Chenming Hu.
 **********/

#include "ngspice.h"
#include <stdio.h>
#include <math.h>
#include "cktdefs.h"
#include "bsim4v3def.h"
#include "sperror.h"


int
BSIM4v3trunc(
GENmodel *inModel,
CKTcircuit *ckt,
double *timeStep)
{
BSIM4v3model *model = (BSIM4v3model*)inModel;
BSIM4v3instance *here;

#ifdef STEPDEBUG
    double debugtemp;
#endif /* STEPDEBUG */

    for (; model != NULL; model = model->BSIM4v3nextModel)
    {    for (here = model->BSIM4v3instances; here != NULL;
	      here = here->BSIM4v3nextInstance)
	 {
	   if (here->BSIM4v3owner != ARCHme) continue;
#ifdef STEPDEBUG
            debugtemp = *timeStep;
#endif /* STEPDEBUG */
            CKTterr(here->BSIM4v3qb,ckt,timeStep);
            CKTterr(here->BSIM4v3qg,ckt,timeStep);
            CKTterr(here->BSIM4v3qd,ckt,timeStep);
            if (here->BSIM4v3trnqsMod)
                CKTterr(here->BSIM4v3qcdump,ckt,timeStep);
            if (here->BSIM4v3rbodyMod)
            {   CKTterr(here->BSIM4v3qbs,ckt,timeStep);
                CKTterr(here->BSIM4v3qbd,ckt,timeStep);
	    }
	    if (here->BSIM4v3rgateMod == 3)
		CKTterr(here->BSIM4v3qgmid,ckt,timeStep);
#ifdef STEPDEBUG
            if(debugtemp != *timeStep)
	    {  printf("device %s reduces step from %g to %g\n",
                       here->BSIM4v3name,debugtemp,*timeStep);
            }
#endif /* STEPDEBUG */
        }
    }
    return(OK);
}
