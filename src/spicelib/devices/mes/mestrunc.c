/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 S. Hwang
**********/
/*
 */

#include "ngspice.h"
#include <stdio.h>
#include "cktdefs.h"
#include "mesdefs.h"
#include "sperror.h"
#include "suffix.h"


int
MEStrunc(inModel,ckt,timeStep)
    GENmodel *inModel;
    register CKTcircuit *ckt;
    double *timeStep;
{
    register MESmodel *model = (MESmodel*)inModel;
    register MESinstance *here;

    for( ; model != NULL; model = model->MESnextModel) {
        for(here=model->MESinstances;here!=NULL;here = here->MESnextInstance){
	    if (here->MESowner != ARCHme) continue;

            CKTterr(here->MESqgs,ckt,timeStep);
            CKTterr(here->MESqgd,ckt,timeStep);
        }
    }
    return(OK);
}
