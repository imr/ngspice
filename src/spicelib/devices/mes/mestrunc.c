/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 S. Hwang
**********/
/*
 */

#include "ngspice.h"
#include "cktdefs.h"
#include "mesdefs.h"
#include "sperror.h"
#include "suffix.h"


int
MEStrunc(GENmodel *inModel, CKTcircuit *ckt, double *timeStep)
{
    MESmodel *model = (MESmodel*)inModel;
    MESinstance *here;

    for( ; model != NULL; model = model->MESnextModel) {
        for(here=model->MESinstances;here!=NULL;here = here->MESnextInstance){
	    if (here->MESowner != ARCHme) continue;

            CKTterr(here->MESqgs,ckt,timeStep);
            CKTterr(here->MESqgd,ckt,timeStep);
        }
    }
    return(OK);
}
