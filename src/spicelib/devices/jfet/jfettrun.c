/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

#include "ngspice.h"
#include <stdio.h>
#include "cktdefs.h"
#include "jfetdefs.h"
#include "sperror.h"
#include "suffix.h"


int
JFETtrunc(inModel,ckt,timeStep)
    GENmodel *inModel;
    register CKTcircuit *ckt;
    double *timeStep;
{
    register JFETmodel *model = (JFETmodel*)inModel;
    register JFETinstance *here;

    for( ; model != NULL; model = model->JFETnextModel) {
        for(here=model->JFETinstances;here!=NULL;here = here->JFETnextInstance){
	    if (here->JFETowner != ARCHme) continue;

            CKTterr(here->JFETqgs,ckt,timeStep);
            CKTterr(here->JFETqgd,ckt,timeStep);
        }
    }
    return(OK);
}
