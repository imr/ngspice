/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Hong J. Park, Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "bsim1def.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int
B1trunc(GENmodel *inModel, CKTcircuit *ckt, double *timeStep)

{
    B1model *model = (B1model*)inModel;
    B1instance *here;
#ifdef STEPDEBUG
    double debugtemp;
#endif /* STEPDEBUG */

    for( ; model != NULL; model = B1nextModel(model)) {
        for(here=B1instances(model);here!=NULL;here = B1nextInstance(here)){

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

