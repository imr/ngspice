/**** BSIM4.4.0  Released by Xuemei (Jane) Xi 03/04/2004 ****/

/**********
 * Copyright 2004 Regents of the University of California. All rights reserved.
 * File: b4trunc.c of BSIM4.4.0.
 * Author: 2000 Weidong Liu
 * Authors: 2001- Xuemei Xi, Jin He, Kanyu Cao, Mohan Dunga, Mansun Chan, Ali Niknejad, Chenming Hu.
 * Project Director: Prof. Chenming Hu.
 **********/

#include "ngspice.h"
#include "cktdefs.h"
#include "bsim4v4def.h"
#include "sperror.h"
#include "suffix.h"

int
BSIM4V4trunc(inModel,ckt,timeStep)
GENmodel *inModel;
CKTcircuit *ckt;
double *timeStep;
{
BSIM4V4model *model = (BSIM4V4model*)inModel;
BSIM4V4instance *here;

#ifdef STEPDEBUG
    double debugtemp;
#endif /* STEPDEBUG */

    for (; model != NULL; model = model->BSIM4V4nextModel)
    {    for (here = model->BSIM4V4instances; here != NULL;
	      here = here->BSIM4V4nextInstance)
	 {
	    if (here->BSIM4V4owner != ARCHme) continue;
#ifdef STEPDEBUG
            debugtemp = *timeStep;
#endif /* STEPDEBUG */
            CKTterr(here->BSIM4V4qb,ckt,timeStep);
            CKTterr(here->BSIM4V4qg,ckt,timeStep);
            CKTterr(here->BSIM4V4qd,ckt,timeStep);
            if (here->BSIM4V4trnqsMod)
                CKTterr(here->BSIM4V4qcdump,ckt,timeStep);
            if (here->BSIM4V4rbodyMod)
            {   CKTterr(here->BSIM4V4qbs,ckt,timeStep);
                CKTterr(here->BSIM4V4qbd,ckt,timeStep);
	    }
	    if (here->BSIM4V4rgateMod == 3)
		CKTterr(here->BSIM4V4qgmid,ckt,timeStep);
#ifdef STEPDEBUG
            if(debugtemp != *timeStep)
	    {  printf("device %s reduces step from %g to %g\n",
                       here->BSIM4V4name,debugtemp,*timeStep);
            }
#endif /* STEPDEBUG */
        }
    }
    return(OK);
}
