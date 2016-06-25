/**** BSIM4.0.0, Released by Weidong Liu 3/24/2000 ****/

/**********
 * Copyright 2000 Regents of the University of California. All rights reserved.
 * File: b4trunc.c of BSIM4.0.0.
 * Authors: Weidong Liu, Kanyu M. Cao, Xiaodong Jin, Chenming Hu.
 * Project Director: Prof. Chenming Hu.
 **********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "bsim4v0def.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
BSIM4v0trunc(inModel,ckt,timeStep)
GENmodel *inModel;
register CKTcircuit *ckt;
double *timeStep;
{
register BSIM4v0model *model = (BSIM4v0model*)inModel;
register BSIM4v0instance *here;

#ifdef STEPDEBUG
    double debugtemp;
#endif /* STEPDEBUG */

    for (; model != NULL; model = model->BSIM4v0nextModel)
    {    for (here = model->BSIM4v0instances; here != NULL;
	      here = here->BSIM4v0nextInstance)
	 {
#ifdef STEPDEBUG
            debugtemp = *timeStep;
#endif /* STEPDEBUG */
            CKTterr(here->BSIM4v0qb,ckt,timeStep);
            CKTterr(here->BSIM4v0qg,ckt,timeStep);
            CKTterr(here->BSIM4v0qd,ckt,timeStep);
            if (here->BSIM4v0trnqsMod)
                CKTterr(here->BSIM4v0qcdump,ckt,timeStep);
            if (here->BSIM4v0rbodyMod)
            {   CKTterr(here->BSIM4v0qbs,ckt,timeStep);
                CKTterr(here->BSIM4v0qbd,ckt,timeStep);
	    }
	    if (here->BSIM4v0rgateMod == 3)
		CKTterr(here->BSIM4v0qgmid,ckt,timeStep);
#ifdef STEPDEBUG
            if(debugtemp != *timeStep)
	    {  printf("device %s reduces step from %g to %g\n",
                       here->BSIM4v0name,debugtemp,*timeStep);
            }
#endif /* STEPDEBUG */
        }
    }
    return(OK);
}
