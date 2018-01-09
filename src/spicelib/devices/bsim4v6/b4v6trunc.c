/**** BSIM4.6.2 Released by Wenwei Yang 07/31/2008 ****/

/**********
 * Copyright 2006 Regents of the University of California. All rights reserved.
 * File: b4trunc.c of BSIM4.6.2.
 * Author: 2000 Weidong Liu
 * Authors: 2001- Xuemei Xi, Mohan Dunga, Ali Niknejad, Chenming Hu.
 * Authors: 2006- Mohan Dunga, Ali Niknejad, Chenming Hu
 * Authors: 2007- Mohan Dunga, Wenwei Yang, Ali Niknejad, Chenming Hu
 * Project Director: Prof. Chenming Hu.
 **********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "bsim4v6def.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
BSIM4v6trunc(
GENmodel *inModel,
CKTcircuit *ckt,
double *timeStep)
{
BSIM4v6model *model = (BSIM4v6model*)inModel;
BSIM4v6instance *here;

#ifdef STEPDEBUG
    double debugtemp;
#endif /* STEPDEBUG */

    for (; model != NULL; model = BSIM4v6nextModel(model))
    {    for (here = BSIM4v6instances(model); here != NULL;
	      here = BSIM4v6nextInstance(here))
	 {

#ifdef STEPDEBUG
            debugtemp = *timeStep;
#endif /* STEPDEBUG */
            CKTterr(here->BSIM4v6qb,ckt,timeStep);
            CKTterr(here->BSIM4v6qg,ckt,timeStep);
            CKTterr(here->BSIM4v6qd,ckt,timeStep);
            if (here->BSIM4v6trnqsMod)
                CKTterr(here->BSIM4v6qcdump,ckt,timeStep);
            if (here->BSIM4v6rbodyMod)
            {   CKTterr(here->BSIM4v6qbs,ckt,timeStep);
                CKTterr(here->BSIM4v6qbd,ckt,timeStep);
	    }
	    if (here->BSIM4v6rgateMod == 3)
		CKTterr(here->BSIM4v6qgmid,ckt,timeStep);
#ifdef STEPDEBUG
            if(debugtemp != *timeStep)
	    {  printf("device %s reduces step from %g to %g\n",
                       here->BSIM4v6name,debugtemp,*timeStep);
            }
#endif /* STEPDEBUG */
        }
    }
    return(OK);
}
