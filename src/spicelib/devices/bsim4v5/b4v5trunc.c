/**** BSIM4.5.0 Released by Xuemei (Jane) Xi 07/29/2005 ****/

/**********
 * Copyright 2004 Regents of the University of California. All rights reserved.
 * File: b4trunc.c of BSIM4.5.0.
 * Author: 2000 Weidong Liu
 * Authors: 2001- Xuemei Xi, Mohan Dunga, Ali Niknejad, Chenming Hu.
 * Project Director: Prof. Chenming Hu.
 **********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "bsim4v5def.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
BSIM4v5trunc(
GENmodel *inModel,
CKTcircuit *ckt,
double *timeStep)
{
BSIM4v5model *model = (BSIM4v5model*)inModel;
BSIM4v5instance *here;

#ifdef STEPDEBUG
    double debugtemp;
#endif /* STEPDEBUG */

    for (; model != NULL; model = BSIM4v5nextModel(model))
    {    for (here = BSIM4v5instances(model); here != NULL;
	      here = BSIM4v5nextInstance(here))
	 {

#ifdef STEPDEBUG
            debugtemp = *timeStep;
#endif /* STEPDEBUG */
            CKTterr(here->BSIM4v5qb,ckt,timeStep);
            CKTterr(here->BSIM4v5qg,ckt,timeStep);
            CKTterr(here->BSIM4v5qd,ckt,timeStep);
            if (here->BSIM4v5trnqsMod)
                CKTterr(here->BSIM4v5qcdump,ckt,timeStep);
            if (here->BSIM4v5rbodyMod)
            {   CKTterr(here->BSIM4v5qbs,ckt,timeStep);
                CKTterr(here->BSIM4v5qbd,ckt,timeStep);
	    }
	    if (here->BSIM4v5rgateMod == 3)
		CKTterr(here->BSIM4v5qgmid,ckt,timeStep);
#ifdef STEPDEBUG
            if(debugtemp != *timeStep)
	    {  printf("device %s reduces step from %g to %g\n",
                       here->BSIM4v5name,debugtemp,*timeStep);
            }
#endif /* STEPDEBUG */
        }
    }
    return(OK);
}
