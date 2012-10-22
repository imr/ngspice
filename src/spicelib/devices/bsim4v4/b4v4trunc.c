/**** BSIM4.4.0  Released by Xuemei (Jane) Xi 03/04/2004 ****/

/**********
 * Copyright 2004 Regents of the University of California. All rights reserved.
 * File: b4trunc.c of BSIM4.4.0.
 * Author: 2000 Weidong Liu
 * Authors: 2001- Xuemei Xi, Jin He, Kanyu Cao, Mohan Dunga, Mansun Chan, Ali Niknejad, Chenming Hu.
 * Project Director: Prof. Chenming Hu.
 **********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "bsim4v4def.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int
BSIM4v4trunc(
GENmodel *inModel,
CKTcircuit *ckt,
double *timeStep)
{
BSIM4v4model *model = (BSIM4v4model*)inModel;
BSIM4v4instance *here;

#ifdef STEPDEBUG
    double debugtemp;
#endif /* STEPDEBUG */

    for (; model != NULL; model = model->BSIM4v4nextModel)
    {    for (here = model->BSIM4v4instances; here != NULL;
	      here = here->BSIM4v4nextInstance)
	 {
#ifdef STEPDEBUG
            debugtemp = *timeStep;
#endif /* STEPDEBUG */
            CKTterr(here->BSIM4v4qb,ckt,timeStep);
            CKTterr(here->BSIM4v4qg,ckt,timeStep);
            CKTterr(here->BSIM4v4qd,ckt,timeStep);
            if (here->BSIM4v4trnqsMod)
                CKTterr(here->BSIM4v4qcdump,ckt,timeStep);
            if (here->BSIM4v4rbodyMod)
            {   CKTterr(here->BSIM4v4qbs,ckt,timeStep);
                CKTterr(here->BSIM4v4qbd,ckt,timeStep);
	    }
	    if (here->BSIM4v4rgateMod == 3)
		CKTterr(here->BSIM4v4qgmid,ckt,timeStep);
#ifdef STEPDEBUG
            if(debugtemp != *timeStep)
	    {  printf("device %s reduces step from %g to %g\n",
                       here->BSIM4v4name,debugtemp,*timeStep);
            }
#endif /* STEPDEBUG */
        }
    }
    return(OK);
}
