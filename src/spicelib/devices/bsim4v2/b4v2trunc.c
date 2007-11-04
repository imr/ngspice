/**** BSIM4.2.1, Released by Xuemei Xi 10/05/2001 ****/

/**********
 * Copyright 2001 Regents of the University of California. All rights reserved.
 * File: b4trunc.c of BSIM4.2.1.
 * Author: 2000 Weidong Liu
 * Authors: Xuemei Xi, Kanyu M. Cao, Hui Wan, Mansun Chan, Chenming Hu.
 * Project Director: Prof. Chenming Hu.
 **********/

#include "ngspice.h"
#include <stdio.h>
#include <math.h>
#include "cktdefs.h"
#include "bsim4v2def.h"
#include "sperror.h"


int
BSIM4v2trunc(inModel,ckt,timeStep)
GENmodel *inModel;
CKTcircuit *ckt;
double *timeStep;
{
BSIM4v2model *model = (BSIM4v2model*)inModel;
BSIM4v2instance *here;

#ifdef STEPDEBUG
    double debugtemp;
#endif /* STEPDEBUG */

    for (; model != NULL; model = model->BSIM4v2nextModel)
    {    for (here = model->BSIM4v2instances; here != NULL;
	      here = here->BSIM4v2nextInstance)
	 {
	 if (here->BSIM4v2owner != ARCHme) continue;
#ifdef STEPDEBUG
            debugtemp = *timeStep;
#endif /* STEPDEBUG */
            CKTterr(here->BSIM4v2qb,ckt,timeStep);
            CKTterr(here->BSIM4v2qg,ckt,timeStep);
            CKTterr(here->BSIM4v2qd,ckt,timeStep);
            if (here->BSIM4v2trnqsMod)
                CKTterr(here->BSIM4v2qcdump,ckt,timeStep);
            if (here->BSIM4v2rbodyMod)
            {   CKTterr(here->BSIM4v2qbs,ckt,timeStep);
                CKTterr(here->BSIM4v2qbd,ckt,timeStep);
	    }
	    if (here->BSIM4v2rgateMod == 3)
		CKTterr(here->BSIM4v2qgmid,ckt,timeStep);
#ifdef STEPDEBUG
            if(debugtemp != *timeStep)
	    {  printf("device %s reduces step from %g to %g\n",
                       here->BSIM4v2name,debugtemp,*timeStep);
            }
#endif /* STEPDEBUG */
        }
    }
    return(OK);
}
