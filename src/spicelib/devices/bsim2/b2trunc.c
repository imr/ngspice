/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Hong J. Park, Thomas L. Quarles
**********/

#include "ngspice.h"
#include "cktdefs.h"
#include "bsim2def.h"
#include "sperror.h"
#include "suffix.h"

int
B2trunc(GENmodel *inModel, CKTcircuit *ckt, double *timeStep)
{
    B2model *model = (B2model*)inModel;
    B2instance *here;
#ifdef STEPDEBUG
    double debugtemp;
#endif /* STEPDEBUG */

    for( ; model != NULL; model = model->B2nextModel) {
        for(here=model->B2instances;here!=NULL;here = here->B2nextInstance){
	    if (here->B2owner != ARCHme) continue;
        
#ifdef STEPDEBUG
            debugtemp = *timeStep;
#endif /* STEPDEBUG */
            CKTterr(here->B2qb,ckt,timeStep);
            CKTterr(here->B2qg,ckt,timeStep);
            CKTterr(here->B2qd,ckt,timeStep);
#ifdef STEPDEBUG
            if(debugtemp != *timeStep) {
                printf("device %s reduces step from %g to %g\n",
                        here->B2name,debugtemp,*timeStep);
            }
#endif /* STEPDEBUG */
        }
    }
    return(OK);
}


