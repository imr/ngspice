/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Hong J. Park, Thomas L. Quarles
**********/

#include "ngspice.h"
#include "cktdefs.h"
#include "bsim1def.h"
#include "sperror.h"
#include "suffix.h"

int
B1trunc(GENmodel *inModel, CKTcircuit *ckt, double *timeStep)

{
    B1model *model = (B1model*)inModel;
    B1instance *here;
#ifdef STEPDEBUG
    double debugtemp;
#endif /* STEPDEBUG */

    for( ; model != NULL; model = model->B1nextModel) {
        for(here=model->B1instances;here!=NULL;here = here->B1nextInstance){
	    if (here->B1owner != ARCHme) continue;

#ifdef STEPDEBUG
            debugtemp = *timeStep;
#endif /* STEPDEBUG */
            CKTterr(here->B1qb,ckt,timeStep);
            CKTterr(here->B1qg,ckt,timeStep);
            CKTterr(here->B1qd,ckt,timeStep);
#ifdef STEPDEBUG
            if(debugtemp != *timeStep) {
                printf("device %s reduces step from %g to %g\n",
                        here->B1name,debugtemp,*timeStep);
            }
#endif /* STEPDEBUG */
        }
    }
    return(OK);
}

